from __future__ import annotations

from ml import train_traffic_accident_v47_backward as backward
from ml.train_best_child_traffic_accident_rate import ROOT


def main() -> None:
    backward.VERSION = 48
    backward.START_VERSION = 47
    backward.OUT = ROOT / "models" / "교통사고" / "41차~49차" / "48차"
    backward.MAX_CV_MAE_INCREASE = 0.02
    backward.PRUNE_WITHIN_TOLERANCE = True
    backward.main()


if __name__ == "__main__":
    main()
