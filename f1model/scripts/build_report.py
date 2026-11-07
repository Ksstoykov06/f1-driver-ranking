"""Injects outputs/report_data.json into report/template.html."""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sens = sys.argv[1] if len(sys.argv) > 1 else ""
h = (R/"report/template.html").read_text(encoding="utf8").replace("/*DATA*/", (R/"outputs/report_data.json").read_text()).replace("/*SENS*/''", json.dumps(sens))
(R/"outputs/f1-model-walkthrough.html").write_text(h, encoding="utf8")
