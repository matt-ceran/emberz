"""LoRA SFT on the Fireworks *serverless Training API* (no dataset upload needed).

  caffeinate -dims callguard/.venv/bin/python callguard/guardian/train_tapi.py --name v1 --end-by 14:15
  ... --resume            # continue from the last saved training state after a crash/network drop

- Sample order is stratified by the target's risk label, so ANY prefix keeps the class balance;
  if the projected end is later than --end-by, the epoch is cut to fit (logged as ev=cut).
- forward_backward/optim_step are pipelined (step i+1 is submitted before step i's results are read).
- Every --ckpt-every steps: a sampler checkpoint (servable/promotable) + a training state (resumable).
Logs: results/tapi_<name>.log.jsonl   State: results/tapi_<name>.state.json   Checkpoints: results/tapi_runs.jsonl
"""
import argparse, collections, datetime as dt, json, math, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "factory"))
import fwclient  # noqa: F401  (loads .env)
import numpy as np
from fireworks.training.sdk import AdamParams, FiretitanServiceClient, types
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = HERE.parents[1]
RES = ROOT / "results"
SERVERLESS = "https://api.fireworks.ai/training/v1/serverless"
DEFAULT_BASE = "accounts/fireworks/models/qwen3p6-35b-a3b"
DEFAULT_HF = "Qwen/Qwen3.6-35B-A3B"


THINK_KW = {"enable_thinking": False}


def encode(tok, msgs, max_len):
    kw = THINK_KW
    p = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True, **kw)
    f = tok.apply_chat_template(msgs, tokenize=True, **kw)
    if hasattr(p, "input_ids"):
        p, f = p["input_ids"], f["input_ids"]
    p, f = list(p), list(f)
    if f[: len(p)] != p or len(f) > max_len:
        return None
    w = [0.0] * len(p) + [1.0] * (len(f) - len(p))
    d = types.Datum(model_input=types.ModelInput.from_ints(f[:-1]),
                    loss_fn_inputs={"target_tokens": types.TensorData.from_numpy(np.array(f[1:], dtype=np.int64)),
                                    "weights": types.TensorData.from_numpy(np.array(w[1:], dtype=np.float32))})
    return d, len(f), int(sum(w))


def load(path, tok, max_len, limit=0):
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if limit and i >= limit:
            break
        msgs = json.loads(line)["messages"]
        e = encode(tok, msgs, max_len)
        if e:
            risk = json.loads(msgs[-1]["content"])["risk"]
            out.append((*e, risk))
    return out


def stratified_order(train, seed):
    """Round-robin across risk classes, proportional to class size, so every prefix is balanced."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for i, x in enumerate(train):
        by[x[3]].append(i)
    for v in by.values():
        rng.shuffle(v)
    n = len(train)
    pos = {k: 0 for k in by}
    order = []
    for j in range(n):
        # pick the class furthest behind its target share
        k = max(by, key=lambda c: (len(by[c]) * (j + 1) / n - pos[c]) if pos[c] < len(by[c]) else -1e9)
        order.append(by[k][pos[k]])
        pos[k] += 1
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--hf", default=DEFAULT_HF)
    ap.add_argument("--train", default=str(ROOT / "data/sft_v1/train.jsonl"))
    ap.add_argument("--dev", default=str(ROOT / "data/sft_v1/dev.jsonl"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--dev-n", type=int, default=96)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--end-by", default="", help="HH:MM local; cut the epoch so it ends by then")
    ap.add_argument("--measure-steps", type=int, default=4)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY_MATT")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-state", default="", help="start from another run's training state (weights + optimizer), e.g. v1-t123")
    ap.add_argument("--thinking-kw", default="enable_thinking", help="chat-template kwarg that turns thinking off ('' for none)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    log_f = (RES / f"tapi_{a.name}.log.jsonl").open("a")
    state_p = RES / f"tapi_{a.name}.state.json"

    def emit(**kv):
        kv["t"] = round(time.time() - t0, 1)
        kv["clock"] = dt.datetime.now().strftime("%H:%M:%S")
        print(json.dumps(kv), flush=True)
        log_f.write(json.dumps(kv) + "\n")
        log_f.flush()

    THINK_KW.clear()
    if a.thinking_kw:
        THINK_KW[a.thinking_kw] = False
    tok = PreTrainedTokenizerFast.from_pretrained(a.hf) if "nemotron" in a.hf.lower() else AutoTokenizer.from_pretrained(a.hf)
    train = load(a.train, tok, a.max_len)
    dev = load(a.dev, tok, a.max_len, a.dev_n)
    order = stratified_order(train, a.seed)
    total = math.ceil(len(train) / a.batch)
    ntok = sum(x[1] for x in train)
    st = json.loads(state_p.read_text()) if (a.resume and state_p.exists()) else None
    max_steps = st["max_steps"] if st else total
    start = st["step"] if st else 0
    emit(ev="data", train=len(train), dev=len(dev), train_tokens=ntok, steps=total, start=start,
         est_cost_usd=round(ntok / 1e6 * 2.14, 2), risk_mix=dict(collections.Counter(x[3] for x in train)))

    svc = FiretitanServiceClient(api_key=os.environ[a.key_env], base_url=SERVERLESS)
    if st:
        tc = svc.create_training_client_from_state_with_optimizer(st["state_path"])
        emit(ev="resumed", from_state=st["state_path"], step=start)
    elif a.init_state:
        tc = svc.create_training_client_from_state_with_optimizer(a.init_state)
        emit(ev="init_from_state", state=a.init_state)
    else:
        tc = svc.create_lora_training_client(base_model=a.base, rank=a.rank)
    emit(ev="session", session=svc.training_session_id, session_name=svc.training_session_name, run=tc.run_id)

    def dev_loss():
        tot, w = 0.0, 0
        for i in range(0, len(dev), a.batch):
            chunk = dev[i:i + a.batch]
            r = tc.forward([x[0] for x in chunk], "cross_entropy").result()
            ls = (r.metrics or {}).get("loss:sum")
            if not ls:  # forward() may not report loss:sum; rebuild it from per-token logprobs
                ls = 0.0
                for (d, _, _, _), out in zip(chunk, getattr(r, "loss_fn_outputs", None) or []):
                    lp = out["logprobs"] if isinstance(out, dict) else getattr(out, "logprobs", None)
                    lp = lp.to_numpy() if hasattr(lp, "to_numpy") else np.asarray(lp if lp is not None else [], dtype=np.float32)
                    wt = d.loss_fn_inputs["weights"]
                    wt = wt.to_numpy() if hasattr(wt, "to_numpy") else np.asarray(wt)
                    if lp.size == wt.size:
                        ls += float(-(lp * wt).sum())
            tot += float(ls)
            w += sum(x[2] for x in chunk)
        return tot / max(1, w)

    if not st:
        emit(ev="dev", step=0, dev_loss=round(dev_loss(), 4))

    def save(step, final=False):
        tag = f"{a.name}-{'fin' if final else 's%03d' % step}"
        sv = tc.save_weights_for_sampler(tag).result().path
        try:
            sp = tc.save_state(f"{a.name}-t{step:03d}").result().path
        except Exception as e:
            sp = None
            emit(ev="save_state_failed", err=str(e)[:200])
        if sp:
            state_p.write_text(json.dumps({"step": step, "state_path": sp, "max_steps": max_steps,
                                           "session": svc.training_session_id, "run": tc.run_id}))
        rec = {"name": a.name, "step": step, "of": max_steps, "base": a.base, "hf": a.hf, "rank": a.rank,
               "sampler_path": sv, "state_path": sp, "session": svc.training_session_id,
               "session_name": svc.training_session_name, "final": final, "ts": int(time.time())}
        with (RES / "tapi_runs.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        emit(ev="checkpoint", step=step, sampler=sv, state=sp)
        return sv

    pending = collections.deque()
    step = start
    measure_t = None
    cut_done = bool(st)
    end_by = None
    if a.end_by:
        h, m = map(int, a.end_by.split(":"))
        end_by = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).timestamp()

    def drain_one():
        s, fbf, opf, wsum, ts = pending.popleft()
        fb = fbf.result()
        opf.result()
        return s, float((fb.metrics or {}).get("loss:sum", 0.0)) / max(1, wsum)

    while step < max_steps:
        idx = order[step * a.batch:(step + 1) * a.batch]
        batch = [train[j] for j in idx]
        lr = a.lr * max(0.1, 1 - step / max(1, max_steps))
        fbf = tc.forward_backward([x[0] for x in batch], "cross_entropy")
        opf = tc.optim_step(AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0))
        pending.append((step + 1, fbf, opf, sum(x[2] for x in batch), time.time()))
        step += 1
        if len(pending) > 1:  # keep one step in flight
            s, loss = drain_one()
            now = time.time()
            if s == start + 1:
                measure_t = now
            if s == start + 1 + a.measure_steps and measure_t:
                sps = (now - measure_t) / a.measure_steps
                eta = now + sps * (max_steps - s)
                emit(ev="speed", s_per_step=round(sps, 1), eta=dt.datetime.fromtimestamp(eta).strftime("%H:%M"),
                     steps_left=max_steps - s)
                if end_by and not cut_done and eta > end_by - 120:  # leave ~2 min for the final save
                    fit = max(s + 1, int(s + (end_by - 120 - now) / sps))
                    if fit < max_steps:
                        emit(ev="cut", from_steps=max_steps, to_steps=fit,
                             samples=fit * a.batch, note="stratified order keeps the class balance")
                        max_steps = fit
                    cut_done = True
            if s % 5 == 0 or s <= 3:
                emit(ev="step", step=s, of=max_steps, lr=round(lr, 7), loss=round(loss, 4),
                     s_per_step=round((now - t0) / max(1, s - start), 1))
            if s % a.ckpt_every == 0 and s < max_steps:
                while pending:
                    drain_one()
                save(s)
    while pending:
        s, loss = drain_one()
        emit(ev="step", step=s, of=max_steps, loss=round(loss, 4))
    sv = save(step, final=True)
    emit(ev="dev", step=step, dev_loss=round(dev_loss(), 4))
    emit(ev="done", sampler=sv, note="session kept for promotion; delete when finished")


if __name__ == "__main__":
    main()
