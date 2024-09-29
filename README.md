# Experiments using score entropy discrete diffusion for protein design 

- This code is a fork of and wholly based on [Aaron Lou's implementation of score entropy discrete diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion), which is fully described in his research paper (which was voted ICML 2024 Best Paper), "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" ([https://arxiv.org/abs/2310.16834](https://arxiv.org/abs/2310.16834))
- The goal of this repository is to evaluate the usefulness of score entropy discrete diffusion for understanding protein sequences
- We hope to be able to use trained models to design new proteins with desired properties 


## Resources  

- Original implementation by Aaron Lou https://github.com/louaaron/Score-Entropy-Discrete-Diffusion 
- Original readme https://github.com/louaaron/Score-Entropy-Discrete-Diffusion 
- Paper link https://arxiv.org/abs/2310.16834


## Experiment 

### Using the existing GPT2 tokenizer

First I just wanted to try using the exact GPT2 tokenizer (with vocab size 50,257) on protein sequences, without any changes. After implementing a simple protein sequence dataset following the existing code, I was surprised to see that after only about 2,000 steps (batch size 256), the model is already producing very realistic protein sequences: 

```
>sample_1
MDTARTIHIMKGKVQGVFFRRAYTRDQARHLGITGWVRNKPDGTVELEAEGPKELLVELLAWCQQGPTARADVDDVDKVIWEPARGIKDFIIR
>sample_2
MAKQCEKIYVYGRVQGVYFRRYTYQRKAQHGITGYAKNLNDVEVLASGQDDVNIKSLMKHWLEHGPPAARVDHVEKTIEYRGRYDSFKIRY
>sample_3
MTDLNRATFLISGLVQGVCFRRASTRDEARRLGVHGWVRNLPDRRVWVLAHEEADVQRLTAWCRKGPPAAKVTEITEREAPGILEGQFLIRGSSDLDRFHVPAG
```

It's impossible to tell much just by looking at the sequence, of course, but [folding these proteins with ESMFold](https://esmatlas.com/resources?action=fold) reveals that they are predicted to fold as expected for this protein family (AcyP), despite being only about 25% sequence identical, which is an amazing result for a generative model. 

![Samples from the SEDD model at 2,000 steps](img/folded.png)


### Using an amino acid tokenizer 

[In progress!]
