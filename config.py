# config.py, single source of truth for all pipeline parameters
# every file imports from here. change a value here, it propagates everywhere
# fill in your own values before running

from dotenv import load_dotenv
import os
load_dotenv()

# ── api keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY             = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL               = "gemini-2.5-flash"
GEMINI_MAX_TOKENS_ANALYST  = 8192   # hypothesis generation, complex structured output
GEMINI_MAX_TOKENS_SUMMARY  = 2048   # narrative summary, short prose
GEMINI_MAX_TOKENS_REC = 2048  # researcher recommendations

# ── data ──────────────────────────────────────────────────────────────────────
SYMBOL               = None # add your trading pair
INTERVAL             = None # add interval
YEARS                = 6
DOLLAR_BAR_THRESHOLD = None      # set your own dollar volume
BARS_PER_YEAR        = None      # derived from your threshold; ~bars/day * 365

# ── triple barrier labeling ───────────────────────────────────────────────────
MAX_HOLD = 20 # or define your own
PT_SL    = None # define your pt sl

# ── cpcv ──────────────────────────────────────────────────────────────────────
N_GROUPS            = 10
K_TEST              = 2
EMBARGO             = MAX_HOLD
STABILITY_THRESHOLD = 0.70

# ── model ─────────────────────────────────────────────────────────────────────
MIN_SAMPLES_LEAF = 20     # more regularization vs 15, less overfit
MIN_AUC          = None # define your own min auc
EDGE_THRESHOLD   = None # define which treshld defines your edge

# ── tuning ────────────────────────────────────────────────────────────────────
N_TRIALS      = 60   # optuna trials; more = better params but slower
N_GROUPS_TUNE = 5    # inner cpcv groups for optuna (smaller = faster)
K_TEST_TUNE   = 2    # inner cpcv test groups

# ── sizing ────────────────────────────────────────────────────────────────────
MIN_PROB    = None   # minimum prob threshold to place a bet
KELLY_FRACS = [0.15, 0.25]
B           = PT_SL[0] / PT_SL[1]   # payout ratio for kelly formula: kelly_f = (p*B - (1-p)) / B
REGIME_GATE_ENABLED = False  # set True to block bets in high-vol hmm regime

# ── cost model ────────────────────────────────────────────────────────────────
FEE_BPS      = 4        # binance taker fee per side
SLIPPAGE_BPS = 3        # estimated market impact per side
CAPITAL      = 5_000  # starting capital in USD for backtest

# ── causal validation ─────────────────────────────────────────────────────────
CAUSAL_P_THRESHOLD = 0.05   # HAC p-value gate for dowhy verdict
HAC_LAGS    = 20     # newey-west lag: ~3 days at 6-7 bars/day
CV_FOLDS   = 5      # cross-fitting folds for robinson estimator
P_THRESHOLD    = 0.05  # significance threshold for dowhy tests


# ── hypothesis generation ─────────────────────────────────────────────────────
N_HYPOTHESES = 2

# ── analysis ──────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
N_BOOTSTRAP   = 2000
N_PERMUTATION = 200

# ── features ──────────────────────────────────────────────────────────────────
# all candidate features (input to cpcv discovery run)
FEATURES = []

# stable features after shap intersection (written by cpcv.py, empty on first run)
FINAL_FEATURES = []

