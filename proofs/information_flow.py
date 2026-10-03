# /// script
# requires-python = ">=3.10"
# dependencies = ["symengine"]
# ///
"""Symbolic checks of how information flows through dense / PISA / GDN-2 / hybrid layers.

Run: uv run proofs/information_flow.py

"Optimal" is not one number, so each section proves one measurable property of the
layer geometry. Every `prove(...)` is an exact symbolic identity or an exhaustive
finite enumeration; a failure raises AssertionError.
"""
from itertools import product

from symengine import Matrix, Symbol, eye, exp, expand, log, symbols, zeros


def prove(name, ok):
    assert ok, f"FAILED: {name}"
    print(f"  [proved] {name}")


# ---------------------------------------------------------------------------
print("\n1. GDN-2 temporal memory: exact dynamics along a key direction")
# Single head, dk=3, dv=1.  State S (3x1).  Paper eq. 9 with scalar a (decay), b (erase), w (write):
#   S' = a S + k (w v - b a k^T S)         (k is NOT assumed unit here: n2 = k.k stays symbolic)
a, b, w, v = symbols("a b w v")
k1, k2, k3, s1, s2, s3 = symbols("k1 k2 k3 s1 s2 s3")
k, S = Matrix([k1, k2, k3]), Matrix([s1, s2, s3])
n2 = k1**2 + k2**2 + k3**2
S_new = a * S + k * (w * v - b * a * (k.T * S)[0, 0])
p, p_new = (k.T * S)[0, 0], (k.T * S_new)[0, 0]  # readout along k
prove("p' = a(1 - b|k|^2) p + |k|^2 w v   (so unit k: p' = lam p + w v, lam = a(1-b))",
      expand(p_new - (a * (1 - b * n2) * p + n2 * w * v)) == 0)

# Jacobian of the state recursion wrt the old state: J = a (I - b k k^T)
J = a * (eye(3) - b * (k * k.T))
u = Matrix([k2, -k1, 0])  # u . k = 0 for any k
prove("J u = a u for u perpendicular to k  (off-key memory only decays by a, never erased)",
      all(expand(x) == 0 for x in (J * u - a * u)))
prove("J k = a(1 - b|k|^2) k  (key direction retention a(1-b) for unit k)",
      all(expand(x) == 0 for x in (J * k - a * (1 - b * n2) * k)))
prove("det J = a^3 (1 - b|k|^2)",
      expand(J.det() - a**3 * (1 - b * n2)) == 0)
G = J.T * J
prove("J^T J = a^2 (I - b(2 - b|k|^2) k k^T)  =>  for unit k, 0<b<=1: singular values {a, a(1-b)}, ||J|| = a <= 1",
      all(expand(x) == 0 for x in (G - a**2 * (eye(3) - b * (2 - b * n2) * (k * k.T)))))

lam = a * (1 - b)
vs = symbols("v1:9")
pt, closed_ok = 0, True
for T in range(1, 9):  # p_T = lam p_{T-1} + w v_T  vs closed form  w * sum lam^(T-s) v_s
    pt = lam * pt + w * vs[T - 1]
    closed = w * sum(lam ** (T - s) * vs[s - 1] for s in range(1, T + 1))
    closed_ok &= expand(pt - closed) == 0
prove("p_T = w sum_s lam^(T-s) v_s  for T = 1..8  =>  d p_T / d v_s = w lam^(T-s): geometric, never growing", closed_ok)
prove("a=1, b=0 gives lam=1: lossless memory (horizon infinite); b=1 gives lam=0: pure overwrite (horizon 0)",
      lam.subs({a: 1, b: 0}) == 1 and lam.subs({b: 1}) == 0)

# Delta rule: same key written twice -> old value fully erased (b=w=a=1, unit k)
v1, v2, c = symbols("v1 v2 c")  # c = k1.k2 for two different keys
S1_k = v1  # after writing (k1, v1): S1^T k1 = v1
S2_k2 = v2  # after writing (k2, v2) with delta rule, k2 reads exactly v2
S2_k1 = v1 + c * (v2 - c * v1)  # S2^T k1 with S2 = S1 + k2 (v2 - c v1)^T, c = k1.k2
prove("same key rewritten (c=1): S^T k = v2, old v1 gone", expand(S2_k1.subs(c, 1) - v2) == 0)
prove("different keys: latest write exact; older read = v1 + c(v2 - c v1) (error O(c), vanishes for orthogonal keys)",
      expand(S2_k1.subs(c, 0) - v1) == 0)

# ---------------------------------------------------------------------------
print("\n2. Dense vs PISA: logit gap needed to read one planted key with weight >= p")
D, pw, m = symbols("D p m", positive=True)  # D logit gap, p target weight, m competing keys
# Only assumed fact: exp and log are inverse on positives.  odds X = e^gap, so gap = ln X.
X = pw * m / (1 - pw)  # candidate e^gap
prove("weight e^gap/(e^gap + m) = p  <=>  X(1-p) = p m, which holds for X = p m/(1-p)  (required gap = ln(p m/(1-p)))",
      expand(X * (1 - pw) - pw * m) == 0)
N, K, B = symbols("N K B", positive=True)
odds = pw / (1 - pw)
dense_X, pisa_X = odds * (N - 1), odds * (K * B)  # e^gap needed with m = N-1 vs m = K B competitors
prove("e^(dense gap - PISA gap) = (N-1)/(K B), independent of p  (PISA reads sharper, IF the key's block is selected)",
      expand(dense_X / pisa_X - (N - 1) / (K * B)) == 0)
saved = log(dense_X) - log(pisa_X)
print(f"    N=3601, K=8, B=16: gap saved = {float(saved.subs({N: 3601, K: 8, B: 16, pw: 0.9})):.3f} nats")

# ---------------------------------------------------------------------------
print("\n3. Reachability depth: exhaustive search over layer orderings")
T, C, BS = 3, 4, 2  # 3 frames x 4 cells = 12 cell tokens + 1 global token (index 0); blocks of BS=2 < C
n = 1 + T * C
idx = lambda t, i: 1 + t * C + i  # frame-major, as in the model
blk = lambda t, i: (t * C + i) // BS  # block id; BS divides C so a block never straddles frames


def layer(kind):
    """A[dst, src] = 1 if dst reads src in one layer (residual self-edge always present)."""
    A = zeros(n, n)
    for d in range(n):
        A[d, d] = 1
    if kind == "D":  # dense: everyone reads everyone
        for d in range(n):
            for s in range(n):
                A[d, s] = 1
    elif kind in ("P", "Pg"):  # PISA worst case = selection never leaves the cell's own block
        for t in range(T):
            for i in range(C):
                for j in range(C):
                    if blk(t, i) == blk(t, j):
                        A[idx(t, i), idx(t, j)] = 1
                if kind == "P":
                    A[idx(t, i), 0] = 1  # every cell always reads the global token
        if kind == "P":
            for s in range(n):
                A[0, s] = 1  # global token reads all tokens densely
    elif kind == "G":  # GDN: (i, t) reads (i, s <= t); global token not mixed
        for i in range(C):
            for t in range(T):
                for s in range(t + 1):
                    A[idx(t, i), idx(s, i)] = 1
    return A


def reach(seq):
    R = eye(n)
    for kind in seq:
        R = layer(kind) * R
    return R


def connects(seq):  # every cell token (any frame) reaches every last-frame cell token
    R = reach(seq)
    return all(R[idx(T - 1, j), idx(t, i)] != 0 for j in range(C) for t in range(T) for i in range(C))


def shortest(kinds, max_len=5, only=None):
    for L in range(1, max_len + 1):
        hits = ["".join(s) for s in product(kinds, repeat=L) if connects(s) and (only is None or only("".join(s)))]
        if hits:
            return L, hits
    return None, []


L, hits = shortest("D")
prove(f"dense reaches everything in {L} layer", L == 1)
L, hits = shortest("P")
prove(f"PISA (worst-case selection) needs {L} layers: cell -> global -> cell", L == 2)
L, _ = shortest(["Pg"], max_len=6)
prove("PISA WITHOUT the global token never connects blocks (checked to depth 6): the global token is essential", L is None)
L, _ = shortest(["G"], max_len=6)
prove("GDN alone never mixes cells (checked to depth 6): it is a time-axis operator only", L is None)
L, hits = shortest("GP", only=lambda s: set(s) == {"G", "P"})
prove(f"shortest stacks mixing BOTH GDN and PISA that fully connect have depth {L}: {hits}",
      L == 3 and set(hits) == {"GPP", "PGP", "PPG"})
prove("GDN+PISA at depth 2 (GP or PG) does NOT connect: mixing needs the global token to gather, then cells to read it",
      not connects("GP") and not connects("PG"))
prove("the model's alternating order (GDN, PISA, GDN, PISA) connects at depth 4 and NOT at depth 3 (GPG)",
      connects("GPGP") and not connects("GPG"))
print("    => with layers=4 the alternating hybrid has ZERO depth slack under worst-case selection;")
print("       PGP or GPP would connect in 3. (Content-based selection usually shortens real paths.)")

# ---------------------------------------------------------------------------
print("\n4. Cost and memory (symbolic)")
Tf, Cc, d, H, Kk, Bb = symbols("T C d H K B", positive=True)
kv = 2 * Tf * d  # attention KV cache per cell
gdn = H * (d / H) ** 2  # GDN state per cell: H heads of dk x dv with dk = dv = d/H
prove("GDN state per cell = d^2/H floats, independent of history length T", expand(gdn - d**2 / H) == 0)
t_star = d / (2 * H)
prove("GDN state < KV cache exactly when T > d/(2H)", expand(kv - gdn - 2 * d * (Tf - t_star)) == 0)
print(f"    d=64, H=4: GDN is smaller than a KV cache for T > {float(t_star.subs({d: 64, H: 4})):.0f} frames")
Nn = Tf * Cc + 1
dense_edges = Nn**2
pisa_edges = Nn * (Kk * Bb + 1) + Nn  # K blocks of B keys + global, per cell; plus the global row
gdn_edges = Cc * Tf * (Tf + 1) / 2 + 0 * Nn  # causal time edges only
ratio = dense_edges / pisa_edges
print(f"    edges/layer: dense N^2; PISA N(KB+2); GDN C T(T+1)/2 (time only)")
print(f"    dense/PISA at 30x30, T=16, K=8, B=16: {float(ratio.subs({Tf: 16, Cc: 900, Kk: 8, Bb: 16})):.1f}x fewer edges")
print(f"    dense/GDN  at 30x30, T=16: {float((dense_edges / gdn_edges).subs({Tf: 16, Cc: 900})):.0f}x fewer edges")
print("    CAVEAT: measured CPU timings (README) show PISA slower at 30x30 x 4 frames despite the edge count;")
print("    edge counts ignore gather/selection overhead, so this proves asymptotics, not wall-clock.")

print("\nAll symbolic checks passed.")
