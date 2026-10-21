# Hishab labelling guideline (v2, 2026-10-21)

One rule per ambiguity we have actually seen in the data. Used by the labellers, by the data
generators, and by anyone hand-labelling. When two rules could apply, the earlier one wins.

v2 changes (decided by the project owner after the v13 relabel spot check): money sent to or
received from family follows its stated purpose, and `gift` is only used when no purpose is
stated; wages I pay are `salary` expense; vehicle rental to the owner ("joma") is `transport`;
sales are split into business (`salary`) and one-off (`other`). The decision order below makes the
existing rules explicit.

## Decision order

1. **Is it a transaction?** (section below). If not, no entry.
2. **type** by direction: did the money leave me (`expense`) or come to me (`income`)?
3. **Loans** by direction (section `type`): borrowing, lending and repaying are always `loan`,
   whatever the money is for.
4. **Stated purpose wins.** If the message says what the money is for, the category is that
   purpose (tables below), whoever paid or received it: bazar → `grocery`, fees → `tuition`,
   doctor → `medical`, rent → `rent`.
5. **No stated purpose:** money between me and family or friends is `gift`; money earned by work or
   business is `salary`; everything else follows the tables.

## What is a transaction

- One entry per amount of money that moved, with its own amount. "lunch 250 r rickshaw 60" is
  two entries.
- **Never emit a stated total.** "lunch 250 r rickshaw 60, mot 310" is two entries, not three.
  A message that only states a total ("ajke 1700 gelo") is one entry with category `other`.
- **No amount, no entry.** "dhar nisi" with no number produces nothing for that item. A message
  with no amounts at all is `[]`.
- Questions, plans and chit-chat are `[]`: "koto khoroch holo?", "kal bazar korbo", "salary
  kobe dibe?". Money I expect but have not received ("tar kache 2k paibo") is a plan: no entry.
- Non-money numbers stay out: "2 kg chal 180" is one entry of 180; "3 jon e 900" is one entry
  of 900 (the split per head is not a transaction).

## type

- `expense`: money left me. `income`: money came to me.
- **Loans by direction**: I borrowed → `loan` income. I lent → `loan` expense. Someone repaid me →
  `loan` income. I repaid → `loan` expense. Installments (EMI, kisti) → `loan` expense.
- **Family and friends, by purpose** (decision order 4 and 5):
  - money I send with a stated purpose takes that purpose's category as an expense:
    "bou ke bazarer jonno 1700 pathalam" → `grocery` expense; "ammar oshudh er jonno 2000" →
    `medical` expense; "cheler school fees pathalam" → `tuition` expense;
  - money they send me with a stated purpose takes that category as income:
    "baba student fees er jonno 10k pathaise" → `tuition` income; "bhai bazarer taka dilo" →
    `grocery` income;
  - no purpose, or plain support, household upkeep ("sonsar er jonno", "family te pathalam"),
    pocket money, salami, presents of money → `gift` (expense or income).
- **Wages I pay** someone (maid, driver, tutor of my child, a worker in my shop) → `salary` expense.
  Wages I receive → `salary` income. (A tutor I pay for my child is `tuition`: the purpose is the
  child's education.)
- Refunds, cashback, change returned, compensation, selling something I owned once → `other`
  income.

## category (expense side)

| what | category |
|---|---|
| eating or drinking anything prepared: restaurant, tea, snacks, sweets, chocolate, toffee, chips, delivery and its delivery charge, canteen, party food, bbq, jilapi, fruit eaten as a snack | `food` |
| raw ingredients and household supplies from bazar/shop/super shop: rice, oil, fish, chicken, vegetables, eggs, spices, salt, soap, detergent, matches, candles | `grocery` |
| house/flat/mess/hostel rent, advance, service charge, seat rent | `rent` |
| electricity, gas, water, wifi/broadband, dish/cable, garbage, holding tax | `utilities` |
| mobile top-up, minute/data/internet packs, bundle offers | `mobile_recharge` |
| rickshaw, CNG, bus, metro, train, launch, ferry, Pathao/Uber, fuel, parking, toll, bike/car servicing and repair, the daily "joma" a driver pays the vehicle owner | `transport` |
| tuition and coaching fees, a private tutor for my child, school/college/university fees, exam/form fees, prospectus, books for study, stationery and teaching material (pens, markers, khata), photocopy, printout, school picnic/trip, lab/material costs | `tuition` |
| doctor, tests, medicine (including a single lozenge or paracetamol), hospital, dental, glasses, vaccination, health insurance | `medical` |
| cinema, concert, game, subscription (Netflix, Spotify, game credits), outing tickets, novels, hobby gear | `entertainment` |
| clothes, shoes, cosmetics, phone and electronics, furniture, household items that are not groceries, presents I buy as objects | `shopping` |
| wages I pay (see `type`) | `salary` |
| anything else: bKash/Nagad charges, bank fees, fines, donations/zakat/sadaqah, money to a beggar, tips, salon/haircut, tailor, laundry, pet, repairs at home | `other` |

Tie-breaks:
- A present that is money → `gift` (unless decision order 4 gives a purpose). A present I buy is
  labelled by what it is: food or sweets to be eaten (a chocolate for a child) → `food`; any other
  object (a toy, a panjabi for abbu) → `shopping`.
- Medicine after a doctor visit → `medical` (not grocery, even if bought at a shop).
- Data/internet pack on a phone → `mobile_recharge`; home broadband bill → `utilities`.
- Fruit or sweets bought to eat now → `food`; bought as groceries for home → `grocery`.
- Books for study → `tuition`; novels for pleasure → `entertainment`; a book with no stated
  purpose → `tuition` (in these messages it is almost always a student's book).
- Stationery for study or teaching → `tuition`; for an office or the house → `shopping`.
- A late fee or fine on a bill → the bill's category (a wifi bill with a late fee is `utilities`);
  a standalone fine (traffic, library) → `other`.

## category (income side)

| what | category |
|---|---|
| job salary, bonus, overtime; freelance and client payments; business sales and profit, including advance payments from customers; tutoring/coaching income; commission; stipend, scholarship, fellowship; ride/delivery earnings and passengers' tips | `salary` |
| money received from family or friends with no stated purpose: pocket money, remittance from abbu/bhai, Eid salami, wedding gift money, prizes | `gift` |
| money received from family or friends for a stated purpose | that purpose's category (see `type`) |
| money borrowed, or a loan repaid to me | `loan` |
| refund, cashback, change returned, compensation, interest, selling my own used things once, insurance payout, found money | `other` |

Tie-breaks:
- Earned → `salary`, even if irregular. Given → `gift` (or the stated purpose). Owed → `loan`.
- **Selling:** regular selling (a shop, the haat, online orders, a farmer's harvest) is business →
  `salary`. Selling my own used things once (an old phone, a charger, a sack of rice from home) →
  `other`.
- A passenger's tip to a driver is earned → `salary`.

## description

Short, in the message's own language and script, naming the thing not the amount: "lunch",
"basha vara", "abbu pathailo", "ওষুধ". Not a sentence, no amount, no "taka".
