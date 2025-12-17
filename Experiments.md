## Evaluation criterion

The system functions as a ranking system as far as I understand, even scores are subjective scores and not real probabilities. But at the same time the
dataset lacks the structure of a ranking dataset (we only have one example for each query, not multiple). And also the design of such dataset would be part
of an evaluation criterion if one wants to obssess over the details, as the data distribution affects the metrics we calcualte.

The other option is classification based options, say cross entropy, accuracy, RoC AUC, etc. They might be a bit useful but I find them also a bit misleading,
also things like standard PR and RoC don't make much sense when you're comparing distributions too. Maybe some sort of kl-divergence will (how much information
are we lacking). I'll try to define some, but as these are distribution dependent, I'm not that positive about these.

I think some kind of correlation, specially spearman can be a useful north star metric here, it captures all the things we care about.

## My thoughts after taking a quick look

General solution architectures

1. Parse the addresses into a structured format, then compare each part
   - How to parse?
     - Rule based parsing, it's a bit hard, for extracting countries I did it but for the rest rules can get quite involved
       - Might take a bit more work than feasible here, specially if we're limited to europe, it shouldn't be that hard to just list every city too
       - Then we only need to parse the postal code, street name, number, which are can be realatively easy
     - NER based
       - Zero shot model for creating a dataset then we can train a smaller model
       - Comparison can be simply a list of levenstein distance of each part, and we can use a simple linear model with basic features to determine the similarity
       - PoC can be done quite easily with the Zero-Shot NER system (it's around 100ms per evaluation, which is quite slow, but we should be able to speed it up somehow)
   - In the parsing approach, we can change the code architecture so that we don't need to parse data from the api, as they already have country/region/building info in a structured format
2. Define a distance function which handles the difference in orderings properly, without parsing the data
   - Of course one can model the problem with a NN, but the failure modes can be quite obscure, a neural network might be a bit more vibe based with the comparison and miss small details in high dimentionality data
   - We can use an embedding based solution, but simple cosine would compare the addresses
     semantically and is not as easy to control/improve, also as dataset is quite small
     training a model to learn the similarity function is not really feasible. although
     I think it's worth a try.
   - We can extract numbers (building and postal code) and then use a simple distance metric to compare the rest, it can be implemented as an incremental change to the existing baseline
   - We can use a slightly better edit distance function, say comparing character ngrams, or maybe sorting before comparing, or some kind of matching/alignment, I don't expect RoI that high with more complex ones, as we can always go the parse based route and that seems a better investment to me if we want to go complex

## Normalization side note

Unicode NFD normaliztion to ensure we're treating diacritics the same way everywhere

## What did I do

Experiments (meaured by spearman score)

1. Baseline + NFD: 0.67
2. Extract country codes with a simple rule: 0.76
3. Extract the house number and postal codE: 0.73 (while can be extracted reliably, the main issue is how to measure the distance)
4. Extract the city and street name using NER: 0.81 (this comes at the hefty cost of 100ms latency for NER inference, but in practice, can be optimized quite a bit by using a simple model, though it's takes a bit to prepare one)
5. Model based probablity calculation: 0.79 (The feature engineering can be quite challenging, also some of the probability numbers are not consistent at all, if an address has a house number and the other does not, the probability of them pointing to the same place is practically zero - if we don't have any context)

I wouldn't deploy this solution to production (because of the slow model) but I think it's a reasonable PoC. In a practical settings honestly I would just use the first result from
the mapbox, as it is quite similar, mapbox probably solves this problem for a living and their ranking should be quite hard to beat. But assuming that's a really business critical issue:

1. Work more on evaluation criterion, mae and spearman don't align, which means quite
   further exploration is needed
1. Create a database from the cities or make a good NER with some properly annotated data, would make the thing faster and more reliable
1. Can we just fix the UX where this data is being entered?
1. Clear up the definition of the ground truth
1. Modularize a bit better, I'm passing in the scorer everywhere
