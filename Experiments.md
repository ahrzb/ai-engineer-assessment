# My thoughts after taking a quick look 

General solution architectures

1. Parse the addresses into a structured format, then compare each part
   * How to parse?
        * Rule based parsing, it's a bit hard, for extracting countries I did it but for the rest rules can get quite involved
            * Might take a bit more work than feasible here, specially if we're limited to europe, it shouldn't be that hard to just list every city too
            * Then we only need to parse the postal code, street name, number, which are can be realatively easy
        * NER based
            * Zero shot model for creating a dataset then we can train a smaller model
            * Comparison can be simply a list of levenstein distance of each part, and we can use a simple linear model with basic features to determine the similarity
            * PoC can be done quite easily with the Zero-Shot NER system (it's around 100ms per evaluation, which is quite slow, but we should be able to speed it up somehow)
    * In the parsing approach, we can change the code architecture so that we don't need to parse data from the api, as they already have country/region/building info in a structured format
2. Define a distance function which handles the difference in orderings properly, without parsing the data
    * Of course one can model the problem with a NN, but the failure modes can be quite obscure, a neural network might be a bit more vibe based with the comparison and miss small details in high dimentionality data
    * We can extract numbers (building and postal code) and then use a simple distance metric to compare the rest, it can be implemented as an incremental change to the existing baseline
    * We can use a slightly better distance function, say comparing character ngrams, or maybe sorting before comparing, or some kind of matching/alignment, I don't expect RoI that high with more complex ones, as we can always go the parse based route and that seems a better investment to me if we want to go complex