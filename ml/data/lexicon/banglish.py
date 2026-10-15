"""Hand-authored Banglish / Bangla lexicon for the programmatic data generator.

Every item carries its category and realistic BDT amount range. Spelling variants are listed
explicitly because phone users never spell the same way twice (tution/tuition, vara/bhara/vhara).
"""

# ---------------------------------------------------------------- items -> (category, lo, hi)
# Each entry: list of surface forms (Banglish), Bangla script forms, category, amount range.
ITEMS = [
    # food
    (["cha", "chaa", "tea"], ["চা"], "food", 10, 60),
    (["singara", "samucha", "samosa"], ["সিঙ্গারা", "সমুচা"], "food", 10, 80),
    (["nasta", "nashta", "breakfast"], ["নাস্তা", "সকালের নাস্তা"], "food", 30, 250),
    (["lunch", "dupurer khabar"], ["দুপুরের খাবার", "লাঞ্চ"], "food", 60, 600),
    (["dinner", "rater khabar"], ["রাতের খাবার", "ডিনার"], "food", 80, 900),
    (["biriyani", "biryani", "kacchi"], ["বিরিয়ানি", "কাচ্চি"], "food", 150, 700),
    (["fuchka", "chotpoti"], ["ফুচকা", "চটপটি"], "food", 30, 150),
    (["burger", "pizza", "fast food"], ["বার্গার", "পিৎজা"], "food", 150, 1500),
    (["coffee", "cold coffee"], ["কফি"], "food", 60, 400),
    (["canteen e khawa", "canteen bill"], ["ক্যান্টিনের খাবার"], "food", 40, 300),
    (["restaurant bill", "restaurant e khawa"], ["রেস্টুরেন্ট বিল"], "food", 300, 4000),
    (["juice", "lassi", "borhani"], ["জুস", "লাচ্ছি", "বোরহানি"], "food", 30, 200),
    (["vat", "bhat", "vat mach"], ["ভাত", "ভাত মাছ"], "food", 40, 300),
    (["roti", "paratha", "porota"], ["রুটি", "পরোটা"], "food", 10, 100),
    (["mishti", "misti", "doi"], ["মিষ্টি", "দই"], "food", 50, 600),
    # grocery
    (["chal", "chaal", "rice"], ["চাল"], "grocery", 300, 4000),
    (["dal", "daal"], ["ডাল"], "grocery", 80, 500),
    (["tel", "soyabin tel", "oil"], ["তেল", "সয়াবিন তেল"], "grocery", 150, 1200),
    (["mach", "maach", "fish"], ["মাছ"], "grocery", 150, 2500),
    (["murgi", "chicken"], ["মুরগি"], "grocery", 150, 1200),
    (["gorur mangsho", "beef", "mangsho"], ["গরুর মাংস", "মাংস"], "grocery", 500, 3000),
    (["dim", "egg"], ["ডিম"], "grocery", 40, 400),
    (["shobji", "sobji", "vegetables"], ["সবজি", "শাকসবজি"], "grocery", 50, 800),
    (["bazar", "bajar", "kacha bazar"], ["বাজার", "কাঁচাবাজার"], "grocery", 200, 6000),
    (["peyaj", "piyaj", "onion"], ["পেঁয়াজ"], "grocery", 40, 500),
    (["dudh", "milk"], ["দুধ"], "grocery", 60, 400),
    (["super shop", "shwapno", "meena bazar"], ["সুপার শপ"], "grocery", 300, 8000),
    (["ata", "moyda", "chini", "lobon"], ["আটা", "ময়দা", "চিনি", "লবণ"], "grocery", 50, 400),
    # rent
    (["basha vara", "basha bhara", "bashar vara", "bari vara", "bari bhara", "house rent", "flat vara"],
     ["বাসা ভাড়া", "বাড়ি ভাড়া", "ফ্ল্যাট ভাড়া"], "rent", 3000, 45000),
    (["room vara", "room bhara", "mess vara", "hostel er vara", "seat vara"],
     ["রুম ভাড়া", "মেসের ভাড়া", "হোস্টেল ভাড়া", "সিট ভাড়া"], "rent", 1500, 9000),
    (["dokan vara", "shop rent", "garage vara"], ["দোকান ভাড়া", "গ্যারেজ ভাড়া"], "rent", 3000, 40000),
    (["advance vara", "advance rent"], ["অ্যাডভান্স ভাড়া"], "rent", 10000, 90000),
    # utilities
    (["current bill", "electricity bill", "bidyut bill", "biddut bill"], ["বিদ্যুৎ বিল", "কারেন্ট বিল"], "utilities", 300, 6000),
    (["gas bill", "gas er bill"], ["গ্যাস বিল"], "utilities", 500, 2000),
    (["pani bill", "wasa bill", "water bill"], ["পানির বিল", "ওয়াসা বিল"], "utilities", 200, 1500),
    (["wifi bill", "net bill", "internet bill", "broadband"], ["ওয়াইফাই বিল", "ইন্টারনেট বিল"], "utilities", 500, 2500),
    (["dish bill", "cable bill"], ["ডিশ বিল"], "utilities", 200, 600),
    (["service charge", "building service charge"], ["সার্ভিস চার্জ"], "utilities", 500, 5000),
    # mobile_recharge
    (["recharge", "mobile recharge", "flexiload", "flexi"], ["মোবাইল রিচার্জ", "রিচার্জ", "ফ্লেক্সিলোড"], "mobile_recharge", 20, 1000),
    (["data pack", "internet pack", "mb kinlam", "gb pack", "net pack"], ["ডাটা প্যাক", "ইন্টারনেট প্যাক"], "mobile_recharge", 30, 1200),
    (["minute pack", "talktime", "bundle"], ["মিনিট প্যাক", "বান্ডেল"], "mobile_recharge", 30, 600),
    # transport
    (["rickshaw", "riksha", "rikshaw", "rickshaw vara", "riksha bhara"], ["রিকশা", "রিকশা ভাড়া"], "transport", 10, 300),
    (["cng", "cng vara", "cng bhara"], ["সিএনজি", "সিএনজি ভাড়া"], "transport", 80, 800),
    (["bus vara", "bus bhara", "bus fare", "local bus"], ["বাস ভাড়া"], "transport", 10, 200),
    (["uber", "pathao", "pathao bike", "uber ride"], ["উবার", "পাঠাও"], "transport", 80, 1500),
    (["train ticket", "bus ticket", "launch ticket"], ["ট্রেনের টিকিট", "বাসের টিকিট", "লঞ্চের টিকিট"], "transport", 100, 3000),
    (["petrol", "octane", "fuel", "gari te tel"], ["পেট্রোল", "অকটেন", "গাড়ির তেল"], "transport", 300, 5000),
    (["auto", "auto vara", "leguna", "tempo"], ["অটো ভাড়া", "লেগুনা"], "transport", 10, 150),
    (["metro", "metro rail", "metro ticket"], ["মেট্রো", "মেট্রো টিকিট"], "transport", 20, 100),
    # tuition (expense side)
    (["tuition fee", "tution fee", "tution fees", "coaching fee", "coaching er beton"],
     ["টিউশন ফি", "কোচিং ফি"], "tuition", 500, 15000),
    (["semester fee", "varsity fee", "university fee", "admission fee"], ["সেমিস্টার ফি", "ভর্তি ফি"], "tuition", 3000, 90000),
    (["exam fee", "form fillup", "form fill up", "registration fee"], ["পরীক্ষার ফি", "ফরম ফিলাপ"], "tuition", 300, 8000),
    (["school fee", "school er beton", "college fee"], ["স্কুলের বেতন", "কলেজ ফি"], "tuition", 500, 12000),
    (["boi", "book", "boi kinlam", "guide boi", "photocopy", "sheet"], ["বই", "গাইড বই", "ফটোকপি", "শিট"], "tuition", 20, 2500),
    # gift
    (["gift", "upohar", "birthday gift"], ["গিফট", "উপহার", "জন্মদিনের গিফট"], "gift", 200, 10000),
    (["salami", "eid salami", "eidi"], ["সালামি", "ঈদ সালামি", "ঈদি"], "gift", 50, 5000),
    (["ammu ke pathalam", "baba ke pathalam", "barite pathalam", "family te pathalam"],
     ["মাকে পাঠালাম", "বাড়িতে পাঠালাম"], "gift", 1000, 30000),
    (["biyer gift", "wedding gift", "biye te dilam"], ["বিয়ের গিফট", "বিয়েতে দিলাম"], "gift", 500, 10000),
    (["donation", "daan", "fitra", "zakat"], ["দান", "ফিতরা", "যাকাত"], "gift", 50, 20000),
    # loan
    (["dhar", "dhaar", "hawlat", "loan"], ["ধার", "হাওলাত", "লোন"], "loan", 200, 50000),
    (["dhar ferot", "dhar shodh", "loan er kisti", "kisti", "emi"], ["ধার ফেরত", "কিস্তি", "লোনের কিস্তি"], "loan", 500, 30000),
    (["bondhu ke dhar dilam", "dosto ke hawlat dilam", "colleague ke dhar dilam"], ["বন্ধুকে ধার দিলাম"], "loan", 200, 20000),
    (["credit card bill", "card er bill"], ["ক্রেডিট কার্ড বিল"], "loan", 1000, 60000),
    # shopping
    (["shirt", "panjabi", "t-shirt", "jeans", "kapor", "jama"], ["শার্ট", "পাঞ্জাবি", "কাপড়", "জামা"], "shopping", 300, 5000),
    (["juta", "shoe", "sandal"], ["জুতা", "স্যান্ডেল"], "shopping", 300, 6000),
    (["headphone", "earphone", "charger", "phone cover"], ["হেডফোন", "চার্জার", "ফোনের কাভার"], "shopping", 100, 4000),
    (["mobile", "phone kinlam", "new phone"], ["মোবাইল", "নতুন ফোন"], "shopping", 8000, 90000),
    (["cosmetics", "makeup", "perfume", "shampoo", "lotion"], ["কসমেটিকস", "পারফিউম", "শ্যাম্পু"], "shopping", 100, 3000),
    (["bag", "backpack", "wallet", "ghori", "watch"], ["ব্যাগ", "ঘড়ি", "ওয়ালেট"], "shopping", 300, 8000),
    (["eid er shopping", "eid shopping", "puja shopping"], ["ঈদের শপিং", "পূজার শপিং"], "shopping", 1000, 30000),
    # medical
    (["doctor fee", "doctor dekhalam", "doctor visit", "chamber fee"], ["ডাক্তার ফি", "ডাক্তার দেখালাম"], "medical", 300, 2500),
    (["medicine", "oshudh", "osud", "dawa", "ousud"], ["ওষুধ", "ঔষধ"], "medical", 50, 5000),
    (["test", "blood test", "x-ray", "ultrasound", "diagnostic"], ["টেস্ট", "ব্লাড টেস্ট", "এক্স-রে"], "medical", 300, 8000),
    (["hospital bill", "clinic bill", "operation"], ["হসপিটাল বিল", "অপারেশন"], "medical", 2000, 150000),
    (["chashma", "glasses", "lens"], ["চশমা"], "medical", 500, 6000),
    (["dentist", "dat er doctor"], ["দাঁতের ডাক্তার"], "medical", 500, 8000),
    # entertainment
    (["cinema", "movie ticket", "star cineplex"], ["সিনেমা", "মুভি টিকিট"], "entertainment", 150, 1000),
    (["netflix", "spotify", "hoichoi", "chorki", "subscription"], ["নেটফ্লিক্স", "সাবস্ক্রিপশন"], "entertainment", 100, 1500),
    (["concert", "concert ticket", "show er ticket"], ["কনসার্ট টিকিট"], "entertainment", 300, 5000),
    (["ghurte gelam", "tour", "picnic", "cox's bazar trip"], ["ঘুরতে গেলাম", "পিকনিক", "ট্যুর"], "entertainment", 500, 30000),
    (["game", "pubg uc", "free fire diamond", "steam game"], ["গেম", "পাবজি ইউসি"], "entertainment", 100, 3000),
    (["cricket match ticket", "football match", "match ticket"], ["ম্যাচের টিকিট"], "entertainment", 200, 3000),
    # other
    (["bkash charge", "cash out charge", "cashout charge", "nagad charge", "send money charge"],
     ["বিকাশ চার্জ", "ক্যাশ আউট চার্জ"], "other", 5, 400),
    (["bank charge", "atm charge", "sms charge"], ["ব্যাংক চার্জ", "এটিএম চার্জ"], "other", 10, 1000),
    (["salon", "chul katlam", "haircut", "parlour"], ["সেলুন", "চুল কাটলাম", "পার্লার"], "other", 50, 2000),
    (["laundry", "kapor dhowa", "iron"], ["লন্ড্রি"], "other", 30, 600),
    (["passport fee", "nid fee", "visa fee"], ["পাসপোর্ট ফি", "ভিসা ফি"], "other", 500, 20000),
    (["vat", "VAT", "tax"], ["ভ্যাট", "ট্যাক্স"], "other", 50, 20000),
    (["jorimana", "fine", "late fee"], ["জরিমানা", "লেট ফি"], "other", 50, 5000),
    (["repair", "phone repair", "bike servicing", "mistri"], ["রিপেয়ার", "মিস্ত্রি", "সার্ভিসিং"], "other", 200, 8000),
    (["kaj er lok", "bua", "driver er beton"], ["বুয়ার বেতন", "ড্রাইভারের বেতন"], "other", 1000, 20000),
    (["courier", "delivery charge", "sundarban courier"], ["কুরিয়ার", "ডেলিভারি চার্জ"], "other", 50, 500),
]

# ---------------------------------------------------------------- income sources
INCOME = [
    (["salary", "beton", "office er salary", "mash er salary"], ["বেতন", "সালারি", "মাসের বেতন"], "salary", 8000, 150000),
    (["bonus", "eid bonus", "festival bonus"], ["বোনাস", "ঈদ বোনাস"], "salary", 3000, 80000),
    (["overtime", "ot er taka"], ["ওভারটাইম"], "salary", 500, 15000),
    (["tuition theke", "tution theke", "tuition er beton", "student er abba dilo", "coaching theke"],
     ["টিউশনি থেকে", "টিউশনির বেতন"], "tuition", 1500, 15000),
    (["freelancing", "upwork", "fiverr", "client dilo", "freelance payment"], ["ফ্রিল্যান্সিং", "ক্লায়েন্ট দিলো"], "other", 2000, 200000),
    (["dhar ferot pelam", "dhar fert dilo", "hawlat ferot", "loan er taka ferot"], ["ধার ফেরত পেলাম", "হাওলাত ফেরত"], "loan", 200, 50000),
    (["dhar nilam", "loan nilam", "bank loan", "hawlat nilam"], ["ধার নিলাম", "লোন নিলাম"], "loan", 1000, 500000),
    (["ammu pathalo", "baba pathalo", "bari theke", "abbu dilo", "mama dilo", "family theke"],
     ["মা পাঠালো", "বাবা পাঠালো", "বাড়ি থেকে"], "gift", 500, 30000),
    (["salami pelam", "eidi pelam", "gift pelam", "birthday te pelam"], ["সালামি পেলাম", "ঈদি পেলাম"], "gift", 100, 10000),
    (["scholarship", "stipend", "brittി", "britti"], ["বৃত্তি", "স্কলারশিপ", "উপবৃত্তি"], "other", 1000, 50000),
    (["old phone becha", "purano laptop bikri", "jinish bikri"], ["পুরানো ফোন বিক্রি"], "other", 1000, 60000),
    (["business e profit", "dokan er income", "sale"], ["দোকানের আয়", "ব্যবসার লাভ"], "other", 1000, 200000),
    (["cashback", "bkash cashback", "offer e pelam"], ["ক্যাশব্যাক"], "other", 10, 500),
    (["prize", "competition e jitlam", "lottery"], ["পুরস্কার", "প্রতিযোগিতায় জিতলাম"], "other", 500, 50000),
]

# ---------------------------------------------------------------- verbs
EXPENSE_VERBS = ["dilam", "disi", "disilam", "diyechi", "dite hoise", "lagse", "laglo", "lagce", "gelo", "geche",
                 "gese", "khoroch holo", "khoros hoise", "kharach korlam", "kinlam", "kinsi", "kinechi", "koira felsi",
                 "pay korlam", "paid korlam", "porse", "porlo", "uthlo", "khaisi", "kheyechi"]
EXPENSE_VERBS_BN = ["দিলাম", "দিয়েছি", "দিতে হলো", "লাগলো", "লেগেছে", "গেলো", "গেছে", "খরচ হলো", "খরচ করলাম",
                    "কিনলাম", "কিনেছি", "পরিশোধ করলাম", "খেলাম"]
INCOME_VERBS = ["pelam", "paisi", "peyechi", "pailam", "dhuklo", "dhukse", "ashlo", "aslo", "ashse", "credited hoise",
                "pathailo", "pathalo", "dilo", "received korlam", "hate ashlo", "account e dhuklo", "bkash e ashlo"]
INCOME_VERBS_BN = ["পেলাম", "পেয়েছি", "ঢুকলো", "ঢুকেছে", "আসলো", "এসেছে", "পাঠালো", "দিলো", "হাতে পেলাম", "একাউন্টে ঢুকলো"]

# ---------------------------------------------------------------- time / connectors / fillers
TIME = ["ajke", "aj", "ajk", "aaj", "kalke", "kal", "gotokal", "gotocal", "shokale", "dupure", "rate", "bikale",
        "ei mashe", "mash er shurute", "mash er sheshe", "shukrobar", "weekend e", "office theke fire", "", "", ""]
TIME_BN = ["আজকে", "আজ", "কাল", "গতকাল", "সকালে", "দুপুরে", "রাতে", "বিকালে", "এই মাসে", "মাসের শেষে", "", ""]
CONNECTORS = [", ", " ar ", " r ", " and ", " tarpor ", " pore ", " sathe ", ", tarpore ", " abar ", " + ", " ; ", " o "]
CONNECTORS_BN = [", ", " আর ", " এবং ", " তারপর ", " পরে ", " সাথে ", ", তারপরে ", " আবার "]
CONTRAST = [" kintu ", " but ", " tobe ", " jodio "]
CONTRAST_BN = [" কিন্তু ", " তবে "]
FILLERS = ["", "", "", "uff ", "ish ", "bhai ", "hmm ", "note: ", "hishab: ", "ajker hishab: ", "khoroch: ",
           "mone rakhte: ", "dost ", "ami ", "amar ", "amader "]
TAIL = ["", "", "", "", ".", " :(", " :)", "...", " !", " lol", " uff", " 😩", " 🙂", " ar ki", " basically", " tai"]
CURRENCY = ["", " taka", " tk", " tk", " taka", "tk", " Tk", " TK", " BDT", "/-", " taka matro", " টাকা"]
CURRENCY_BN = [" টাকা", " টাকা", " টাকায়", " টাকার", " ৳", ""]

# No-transaction sentences (output [])
NO_TX = [
    "Ajke kichu khoroch kori nai", "aj kono kharach nai", "Gotokal shudhu barite chilam", "Ajke bazar e jai nai",
    "mash er sheshe hishab korbo", "kal salary pabo hopefully", "Ajke ki ki kinbo list korchi", "tuition e jete parini",
    "ei mashe khoroch komate hobe", "bkash e balance check korlam", "ammu ke phone dilam", "aj office off",
    "Rickshaw paini, hete gesi", "dokan bondho chilo, kichu kinte parini", "আজ কিছু খরচ হয়নি", "আজকে বাজারে যাইনি",
    "কালকে বেতন পাবো", "আজ সারাদিন বাসায় ছিলাম", "খরচের হিসাব পরে করবো", "আজ কোনো টাকা দেইনি",
    "Today I didn't spend anything", "No expenses today", "ajke free chilam, kono kharach nai", "hishab app ta valo",
    "ami ki ki kharach korsi mone nai", "bkash e taka nai, kal pathabo", "kal tuition er beton pabo", "porikkha ache tai bashay",
    "Shob bill next week e dibo", "net er bill ekhono dei nai", "ammu bollo bazar korte, kori nai", "dhar ta kal ferot dibo",
]

# Spelled-out numbers. Real messages write "ponero hajar" / "পনেরো হাজার" / "dui lakh";
# the v4 Qwen model read "পনেরো হাজার" as 7000, so the generator now emits these too.
NUM_WORDS_BN = {
    1: "এক", 2: "দুই", 3: "তিন", 4: "চার", 5: "পাঁচ", 6: "ছয়", 7: "সাত", 8: "আট", 9: "নয়", 10: "দশ",
    11: "এগারো", 12: "বারো", 13: "তেরো", 14: "চৌদ্দ", 15: "পনেরো", 16: "ষোলো", 17: "সতেরো", 18: "আঠারো",
    19: "উনিশ", 20: "বিশ", 25: "পঁচিশ", 30: "ত্রিশ", 35: "পঁয়ত্রিশ", 40: "চল্লিশ", 45: "পঁয়তাল্লিশ", 50: "পঞ্চাশ",
    60: "ষাট", 70: "সত্তর", 75: "পঁচাত্তর", 80: "আশি", 90: "নব্বই", 100: "একশ", 150: "দেড়শ", 200: "দুইশ",
    250: "আড়াইশ", 300: "তিনশ", 400: "চারশ", 500: "পাঁচশ", 600: "ছয়শ", 700: "সাতশ", 800: "আটশ", 900: "নয়শ",
}
NUM_WORDS_ROMAN = {
    1: ["ek"], 2: ["dui"], 3: ["tin"], 4: ["char"], 5: ["pach", "panch"], 6: ["choy", "chhoy"], 7: ["sat"],
    8: ["at", "aat"], 9: ["noy"], 10: ["dosh"], 11: ["egaro"], 12: ["baro"], 13: ["tero"], 14: ["chouddo"],
    15: ["ponero", "pnero"], 16: ["sholo"], 17: ["sotero"], 18: ["atharo"], 19: ["unish"], 20: ["bish", "bis"],
    25: ["pochish"], 30: ["trish", "tirish"], 35: ["poytrish"], 40: ["chollish"], 45: ["poytallish"],
    50: ["ponchash", "pochash"], 60: ["shat"], 70: ["sottor"], 75: ["pochattor"], 80: ["ashi"], 90: ["nobboi"],
    100: ["eksho", "ek sho"], 150: ["dersho", "der sho"], 200: ["duisho", "dui sho"], 250: ["araisho", "arai sho"],
    300: ["tinsho"], 400: ["charsho"], 500: ["pachsho", "panch sho"], 600: ["choysho"], 700: ["satsho"],
    800: ["atsho"], 900: ["noysho"],
}
# "dersho"=150, "arai hajar"=2500, "der lakh"=150000 style fractions.
FRACTION_WORDS = {1.5: (["der"], "দেড়"), 2.5: (["arai", "aarai"], "আড়াই"), 3.5: (["sare tin"], "সাড়ে তিন"),
                  4.5: (["sare char"], "সাড়ে চার"), 5.5: (["sare pach"], "সাড়ে পাঁচ"), 7.5: (["sare sat"], "সাড়ে সাত")}
