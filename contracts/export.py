"""Write JSON Schema for every contract to contracts/schemas/."""
import json
from pathlib import Path

from contracts.models import ALL_MODELS

out = Path(__file__).parent / "schemas"
out.mkdir(exist_ok=True)
for name, model in ALL_MODELS.items():
    (out / f"{name}.json").write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
    print("wrote", name)
