Here is a clear, business-focused explanation you can share directly with non-technical stakeholders or use in a presentation:

---

### The Business Challenge: Empty Rooms vs. Overbooking Penalties
Across our two properties — **Downtown Hotel (124 rooms)** and **Seaside Hotel (103 rooms)** — we face a classic hospitality trade-off every time a booking request arrives:
* **Roughly 25%–40% of guests cancel** before check-in. If we stop taking bookings as soon as we hit 100% of our physical rooms, cancellations will leave rooms sitting empty.
* On the other hand, if we accept too many bookings — or fill the hotel months in advance with cheap bargain rates — we run out of rooms at check-in. Turning away ("walking") an arriving guest earns **zero revenue** and costs **2.5x their entire stay value** in penalties and relocation costs.

### How Our System Works: 3 Questions for Every Booking
Whenever a new reservation request comes in, our system ([solution.py](file:///Users/homefolder/Downloads/projects/hackathon2/solution.py)) evaluates it in less than two milliseconds by asking three practical business questions:

#### 1. *"How likely is this guest to actually show up?"* (Reliability Forecast)
Not all bookings are equal. Using patterns learned from over 78,000 past stays ([train.csv](file:///Users/homefolder/Downloads/projects/hackathon2/train.csv)), the system looks at 25 details of the request — such as how far ahead they are booking, booking channel, travel agent, room type, party size, and whether the guest has cancelled with us before — to estimate a **show-up probability** (0% to 100%).
* Instead of counting every booking as taking up 1 full room, a guest with an 80% chance of arriving is counted as **0.8 of a room** on our calendar.
* The model emphasizes recent guest trends and cross-checks two forecasting views so sudden shifts in international traveler demographics don't skew its estimates.

#### 2. *"Do we have safe room capacity for every night of their stay?"* (Controlled Overbooking)
Because we know a predictable fraction of guests will cancel, the system tracks our **expected room count** for every night of the year and allows a carefully calibrated safety buffer (**up to 119% of physical rooms at Downtown and 120% at Seaside**).
* If any night of a requested stay is already at its safety ceiling, the system turns the booking down to protect the hotel from walk penalties (with a narrow exception on the arrival night only for ultra-premium rates that more than cover displacement risk).
* As rooms approach full capacity, the system also automatically raises its minimum acceptable rate to add an extra cushion against last-minute overflow.

#### 3. *"Should we take this rate today, or save the room for a higher-paying guest later?"* (Dynamic Hurdle Price)
Having an empty room today doesn't mean we should sell it cheap if better demand is still on the way.
* Looking at historical demand for that specific **hotel, month, and day of the week**, the system knows how many high-paying guests typically book within the remaining days before check-in.
* From this, it sets a **dynamic minimum price ("hurdle rate")** for each night based on **how many rooms are left** and **how much time remains before arrival**:
  * **Far in advance or when few rooms remain**: The hurdle price is high — we hold out for premium guests.
  * **Close to check-in with open rooms**: The hurdle price drops automatically so rooms don't go unsold.
* For multi-night stays, the system weighs both the average nightly hurdle across the stay and the busiest "bottleneck" night. If the guest's nightly rate (adjusted for their reliability) clears that bar, we **Accept**; otherwise, we **Decline** and save the inventory.

---

### Bottom-Line Impact on the 2017 Hold-Out ([evaluate.py](file:///Users/homefolder/Downloads/projects/hackathon2/evaluate.py))
* **Accepting all requests (`AcceptAll`)**: **EUR -2,008,559** (severe overbooking penalties wipe out room revenue).
* **Taking no new requests (`RejectAll`)**: **EUR 2,441,036** (from pre-existing commitments alone).
* **Our Recommendation System (`Solution`)**: **EUR 5,560,407** — generating **+EUR 3.12M** in additional net profit (**2.28x** the baseline) by filling rooms with reliable, high-value guests while keeping walk penalties tightly contained.