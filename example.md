

Here are two real 2017 requests from [test.csv](file:///Users/homefolder/Downloads/projects/hackathon2/test.csv) traced step-by-step through `Solution.decide()`:

---

### 1. Accepted Example — `booking_id = 12095`
- **Request:** Seaside Hotel · Arrival `2017-06-13` · `2` nights · `days_ahead = 162` · Rate $r = €185.00$ (`Online TA`, `No Deposit`)
- **Step 1 (Show-up probability):** $p = 0.8320 \ge 0.05$ ✓
- **Step 2 (Capacity check):**
  - Free rooms on those two nights: $\text{cap} = [103, 103]$, cap multiplier $K_h = 1.20 \implies$ virtual limit $123.6$ rooms/night.
  - Expected load before: $[94.36, 95.70]$ $\to$ after: $E^{\text{next}} = [95.20, 96.53]$ (loads $92.42\%, 93.72\% \le 120\%$) ✓
- **Step 3 (Nightly bids $b_k$):**
  - Since both loads are below $K_{\text{soft}} = 95\%$, the walk-risk floor is $€0.00$.
  - Historical future-demand bids: $b = [€138.28,\; €136.75]$.
- **Step 4 (Stay hurdle $\lambda$):**
  $$\lambda = 0.70 \times \frac{138.28 + 136.75}{2} + 0.30 \times 138.28 = €137.74$$
- **Step 5 (Risk-adjusted rate vs. hurdle):**
  $$\text{eff\_rate} = 185.00 \times 0.8320^{0.35} = €173.47$$
  Since $€173.47 \ge €137.74$, the system **Accepts (`True`)** and updates expected occupancy to $[95.20, 96.53]$.

---

### 2. Rejected Example — `booking_id = 62158`
- **Request:** Downtown Hotel · Arrival `2017-01-03` · `2` nights · `days_ahead = 2` · Rate $r = €88.00$ (`Online TA`, `No Deposit`)
- **Step 1 (Show-up probability):** $p = 0.7403 \ge 0.05$ ✓
- **Step 2 (Capacity check):**
  - Free rooms (after 2016 in-house guests): $\text{cap} = [96, 110]$, $K_h = 1.19 \implies$ virtual limits $[114.24, 130.90]$.
  - Expected load before: $[113.42, 124.14]$ $\to$ after: $E^{\text{next}} = [114.16, 124.88]$ (loads $118.91\%, 113.53\% \le 119\%$) ✓
- **Step 3 (Nightly bids $b_k$):**
  - Loads exceed $95\%$, activating walk-risk floors of $[€59.78,\; €46.33]$.
  - Historical future-demand bids are higher: $[€113.95,\; €93.80]$, so $b = [€113.95,\; €93.80]$.
- **Step 4 (Stay hurdle $\lambda$):**
  $$\lambda = 0.70 \times \frac{113.95 + 93.80}{2} + 0.30 \times 113.95 = €106.90$$
- **Step 5 (Risk-adjusted rate vs. hurdle):**
  $$\text{eff\_rate} = 88.00 \times 0.7403^{0.35} = €79.21$$
  Since $€79.21 < €106.90$, the system **Rejects (`False`)** to avoid locking up nearly full nights for a low-rate booking.
