from dataclasses import dataclass
from datetime import datetime, timedelta

DATE_FMT = "%Y-%m-%d"

@dataclass(frozen=True)
class Phase:
    number: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

def _fmt(dt):
    return dt.strftime(DATE_FMT)

def build_paper_phases(
    start_train="2016-01-01",
    start_val="2016-11-01",
    start_test="2017-01-01",
    final_test_end="2022-05-01",
    step_days=163,
):
    # Mirrors the released ESTIMATE rolling schedule:
    # train Jan-Oct 2016, validation Nov-Dec 2016,
    # first backtest begins Jan 1 2017, then shift 163 calendar days.
    train0 = datetime.strptime(start_train, DATE_FMT)
    val0 = datetime.strptime(start_val, DATE_FMT)
    test0 = datetime.strptime(start_test, DATE_FMT)
    final_end = datetime.strptime(final_test_end, DATE_FMT)

    train_len = val0 - train0
    val_len = test0 - val0

    phases = []
    phase_num = 1
    test_s = test0
    while test_s < final_end:
        test_e = min(test_s + timedelta(days=step_days), final_end)
        shift = test_s - test0
        tr_s = train0 + shift
        va_s = val0 + shift
        phases.append(Phase(
            number=phase_num,
            train_start=_fmt(tr_s),
            train_end=_fmt(tr_s + train_len - timedelta(days=1)),
            val_start=_fmt(va_s),
            val_end=_fmt(va_s + val_len - timedelta(days=1)),
            test_start=_fmt(test_s),
            test_end=_fmt(test_e),
        ))
        test_s = test_e
        phase_num += 1
    return phases
