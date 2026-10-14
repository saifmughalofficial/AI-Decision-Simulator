"""Quick self-test:  python tests/test_core.py   (no Polars/Streamlit needed)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_sim import explain, llm
from decision_sim.data_gen import generate
from decision_sim.engine import Engine
from decision_sim.understand import rule_parse, understand


def main():
    eng = Engine(generate(400))
    fc = eng.forecast("career", "act_now", "positive", "low", "weeks")
    assert 0 <= fc.lo <= fc.p <= fc.hi <= 1 and fc.confidence in ("Low", "Medium", "High")
    cmp = eng.compare("career", "positive", "low", "weeks")
    best = eng.best_choice(cmp)
    assert best in cmp
    assert "Similar past decisions" in explain.build(
        {"choice": "act_now", "previous_experience": "positive", "risk_level": "low", "time_available": "weeks"},
        fc, cmp, best, [])
    assert rule_parse("Should I quit my job? it's risky")["risk_level"] == "high"
    assert not understand("hello there", [], use_llm=False)["is_decision"]
    assert llm.extract_json('noise {"a": 1} tail') == {"a": 1}
    adv = llm.normalize_advice({"is_decision": True, "recommendation": "Wait",
                                "options": [{"name": "Go", "score": 140}, {"name": "Wait", "score": "62"}]})
    assert adv["recommendation"] == "Wait" and adv["options"][0]["score"] == 100 and adv["rec_score"] == 62
    assert llm.normalize_advice({"is_decision": True, "options": []}) is None
    bt = eng.backtest(150)
    assert bt["accuracy"] > bt["baseline"] - 0.02, bt
    print("All tests passed.", bt)


if __name__ == "__main__":
    main()
