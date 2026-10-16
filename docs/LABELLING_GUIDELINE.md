# Hishab labelling guideline (v1, 2026-10-16)

One rule per ambiguity we have actually seen in the data. Used by the Codex judge, by the
data generators, and by anyone hand-labelling. When two rules could apply, the earlier one wins.

## What is a transaction

- One entry per amount of money that moved, with its own amount. "lunch 250 r rickshaw 60" is
  two entries.
- **Never emit a stated total.** "lunch 250 r rickshaw 60, mot 310" is two entries, not three.
  A message that only states a total ("ajke 1700 gelo") is one entry with category `other`.
- **No amount, no entry.** "dhar nisi" with no number produces nothing for that item. A message
  with no amounts at all is `[]`.
- Questions, plans and chit-chat are `[]`: "koto khoroch holo?", "kal bazar korbo", "salary
  kobe dibe?".
- Non-money numbers stay out: "2 kg chal 180" is one entry of 180; "3 jon e 900" is one entry
  of 900 (the split per head is not a transaction).

## type

- `expense`: money left me. `income`: money came to me.
- **Loans by direction**: I borrowed → `loan` income. I lent → `loan` expense. Someone repaid me →
  `loan` income. I repaid → `loan` expense. Installments (EMI, kisti) → `loan` expense.
- Money I sent to family or friends as support/upohar → `gift` expense. Money they sent me →
  `gift` income.
- Refunds, cashback, selling something I owned → `other` income.

## category (expense side)

| what | category |
|---|---|
| eating or drinking anything prepared: restaurant, tea, snacks, delivery, canteen, party food, bbq, jilapi, fruit eaten as a snack | `food` |
| raw ingredients and household supplies from bazar/shop/super shop: rice, oil, fish, chicken, vegetables, eggs, spices, soap, detergent | `grocery` |
| house/flat/mess/hostel rent, advance, service charge, seat rent | `rent` |
| electricity, gas, water, wifi/broadband, dish/cable, garbage, holding tax | `utilities` |
| mobile top-up, minute/data/internet packs, bundle offers | `mobile_recharge` |
| rickshaw, CNG, bus, metro, train, launch, Pathao/Uber, fuel, parking, toll, bike/car servicing and repair | `transport` |
| tuition and coaching fees, school/college/university fees, exam/form fees, books, stationery, photocopy, printout, school trip, lab/material costs | `tuition` |
| doctor, tests, medicine, hospital, dental, glasses, vaccination, health insurance | `medical` |
| cinema, concert, game, subscription (Netflix, Spotify, game credits), outing tickets, hobby gear | `entertainment` |
| clothes, shoes, cosmetics, phone and electronics, furniture, household items that are not groceries, gifts bought as objects | `shopping` |
| anything else: bKash/Nagad charges, bank fees, fines, donations/zakat/sadaqah, tips, salon/haircut, laundry, pet, repairs at home | `other` |

Tie-breaks:
- A gift that is money → `gift`; a gift that is an object I bought → `shopping`.
- Medicine after a doctor visit → `medical` (not grocery, even if bought at a shop).
- Data/internet pack on a phone → `mobile_recharge`; home broadband bill → `utilities`.
- Fruit or sweets bought to eat now → `food`; bought as groceries for home → `grocery`.
- Books for study → `tuition`; novels for pleasure → `entertainment`.

## category (income side)

| what | category |
|---|---|
| job salary, bonus, overtime; freelance and client payments; business sales and profit; tutoring/coaching income; stipend, scholarship, fellowship; ride/delivery earnings | `salary` |
| money received from family or friends: pocket money, remittance from abbu/bhai, Eid salami, wedding gift money, prizes | `gift` |
| money borrowed, or a loan repaid to me | `loan` |
| refund, cashback, interest, selling old things, insurance payout, found money | `other` |

Tie-breaks:
- Earned → `salary`, even if irregular. Given → `gift`. Owed → `loan`.
- A passenger's tip to a driver is earned → `salary`.

## description

Short, in the message's own language and script, naming the thing not the amount: "lunch",
"basha vara", "abbu pathailo", "ওষুধ". Not a sentence, no amount, no "taka".
