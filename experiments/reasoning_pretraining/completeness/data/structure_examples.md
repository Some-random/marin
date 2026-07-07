# EntailmentBank-v3

# QASC

## ex0
```json
{
  "id": "3E7TUJ2EGCLQNOV1WEAJ2NN9ROPD9K",
  "question": "What type of water formation is formed by clouds?",
  "choices": {
    "text": [
      "pearls",
      "streams",
      "shells",
      "diamonds",
      "rain",
      "beads",
      "cooled",
      "liquid"
    ],
    "label": [
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H"
    ]
  },
  "answerKey": "F",
  "fact1": "beads of water are formed by water vapor condensing",
  "fact2": "Clouds are made of water vapor.",
  "combinedfact": "Beads of water can be formed by clouds.",
  "formatted_question": "What type of water formation is formed by clouds? (A) pearls (B) streams (C) shells (D) diamonds (E) rain (F) beads (G) cooled (H) liquid"
}
```

## ex1
```json
{
  "id": "3LS2AMNW5FPNJK3C3PZLZCPX562OQO",
  "question": "Where do beads of water come from?",
  "choices": {
    "text": [
      "Too much water",
      "underground systems",
      "When the water is too cold",
      "Water spills",
      "Vapor turning into a liquid",
      "Warm air moving into cold air",
      "At the peak of a mountain",
      "To another location like underground"
    ],
    "label": [
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H"
    ]
  },
  "answerKey": "E",
  "fact1": "beads of water are formed by water vapor condensing",
  "fact2": "Condensation is the change of water vapor to a liquid.",
  "combinedfact": "Vapor turning into a liquid leaves behind beads of water",
  "formatted_question": "Where do beads of water come from? (A) Too much water (B) underground systems (C) When the water is too cold (D) Water spills (E) Vapor turning into a liquid (F) Warm air moving into cold air (G) At the peak of a mountain (H) To another location like underground"
}
```

## ex2
```json
{
  "id": "3TMFV4NEP8DPIPCI8H9VUFHJG8V8W3",
  "question": "What forms beads of water? ",
  "choices": {
    "text": [
      "Necklaces.",
      "Steam.",
      "Glass beads .",
      "a wave",
      "tiny",
      "a solute",
      "rain",
      "Bracelets."
    ],
    "label": [
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H"
    ]
  },
  "answerKey": "B",
  "fact1": "beads of water are formed by water vapor condensing",
  "fact2": "An example of water vapor is steam.",
  "combinedfact": "Steam forms beads of water.",
  "formatted_question": "What forms beads of water?  (A) Necklaces. (B) Steam. (C) Glass beads . (D) a wave (E) tiny (F) a solute (G) rain (H) Bracelets."
}
```

# ProofWriter

## ex0
```json
{
  "id": "AttNeg-OWA-D0-4611",
  "maxD": 0,
  "NFact": 7,
  "NRule": 8,
  "theory": "Gary is furry. Gary is nice. Gary is red. Gary is rough. Gary is not smart. Gary is white. Gary is young. If Gary is nice and Gary is not white then Gary is red. If someone is white then they are red. All young people are furry. If someone is white and not red then they are furry. Smart, red people are rough. If Gary is not red and Gary is not furry then Gary is not smart. If Gary is white then Gary is not smart. If someone is rough and not white then they are not smart.",
  "question": "Gary is white.",
  "answer": "True",
  "QDep": 0,
  "QLen": 1.0,
  "allProofs": "@0: Gary is furry.[(triple1 OR ((triple7) -> rule3))] Gary is nice.[(triple2)] Gary is not smart.[(triple5 OR ((triple6) -> rule7))] Gary is red.[(triple3 OR ((triple6) -> rule2))] Gary is rough.[(triple4)] Gary is white.[(triple6)] Gary is young.[(triple7)]",
  "config": "depth-0"
}
```

## ex1
```json
{
  "id": "AttNeg-OWA-D0-4611",
  "maxD": 0,
  "NFact": 7,
  "NRule": 8,
  "theory": "Gary is furry. Gary is nice. Gary is red. Gary is rough. Gary is not smart. Gary is white. Gary is young. If Gary is nice and Gary is not white then Gary is red. If someone is white then they are red. All young people are furry. If someone is white and not red then they are furry. Smart, red people are rough. If Gary is not red and Gary is not furry then Gary is not smart. If Gary is white then Gary is not smart. If someone is rough and not white then they are not smart.",
  "question": "Gary is not nice.",
  "answer": "False",
  "QDep": 0,
  "QLen": 1.0,
  "allProofs": "@0: Gary is furry.[(triple1 OR ((triple7) -> rule3))] Gary is nice.[(triple2)] Gary is not smart.[(triple5 OR ((triple6) -> rule7))] Gary is red.[(triple3 OR ((triple6) -> rule2))] Gary is rough.[(triple4)] Gary is white.[(triple6)] Gary is young.[(triple7)]",
  "config": "depth-0"
}
```

## ex2
```json
{
  "id": "AttNoneg-OWA-D0-3321",
  "maxD": 0,
  "NFact": 3,
  "NRule": 3,
  "theory": "Bob is furry. Dave is quiet. Harry is big. Rough things are white. All rough things are white. If Dave is rough then Dave is green.",
  "question": "Bob is furry.",
  "answer": "True",
  "QDep": 0,
  "QLen": 1.0,
  "allProofs": "@0: Bob is furry.[(triple1)] Dave is quiet.[(triple2)] Harry is big.[(triple3)]",
  "config": "depth-0"
}
```

