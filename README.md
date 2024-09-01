# Mapping HPE Representations

## Task formulation
Given Dataset D1, D2, ..., Dn and representations R1, R2, ..., Rn, find if mapping exists between representations (i,j) <- (1..n) Ri <-> Rj. Or find good enough representations for each Di so that we can create one whole dataset with all the representations for each image.

<hr>

## Important links

[https://material-cowbell-938.notion.site/CMP-b2ce1070ae064165b11c3c81824d0f67?pvs=74] - Stores important logs and small summary of configurations, modifications, trainings, and results.

[https://github.com/bilguudeiblgd/mapping-HPE-representations/tree/main] - Analyzing keypoint differences, translational model from representations.

[https://github.com/bilguudeiblgd/ViTPose/tree/combined_joint_prediction_merged] - Vitpose base training&validation for COCO&MPII&AIC(Union).

[https://github.com/bilguudeiblgd/ViTPose/tree/combined_joint_prediction] - Vitpose base training&validation for COCO&MPII (concat).

[https://github.com/bilguudeiblgd/ViTPose/tree/hack_vitpose_base] - Hacking vitpose base, with multiple decoders for each dataset.

[https://github.com/bilguudeiblgd/ViTPose/tree/hack_vitpose_plus] - Hacking vitpose+ base. with multiple decords, it's easier as detector code is already written. 

[https://github.com/bilguudeiblgd/ViTPose/tree/run_vitpose_coco_on_mpii] - Run coco model on mpii (config)

[https://github.com/bilguudeiblgd/ViTPose/tree/run_mpii_model_on_coco] - Run mpii model on coco (config)

<hr>

## Research summary & explanations 
We tried 2 directions. Direction 1 involves translating representations directly, with mostly simple neural nets. The code is in [https://github.com/bilguudeiblgd/mapping-HPE-representations/tree/main].  Direction 2 uses ViTPose to come up with multiple representations in one model. The code mostly uses main ViTPose code and modification of it on my fork, which is given above in #important-links.

<hr>

### Direction 1.
Our first idea was to try training a small net to see if translation between representations is possible. We mainly worked with only COCO and MPII as most of the other datasets are very close to them. 

The issue is to have 2 representations on one dataset so we can train between them. Firstly, I ran COCO on MPII(the config is given above) to get COCO keypoint representation of MPII dataset. From then, we treat that X and Y as MPII groundtruth. From here, the experiment is in the file [https://github.com/bilguudeiblgd/mapping-HPE-representations/blob/main/translation_model_coco2mpii.ipynb]. The experiment, in my opinion, was quite successful. The better the X(COCO prediction on MPII image), the better the translation. With close to 78 AP COCO model, the translation achieves 90+ PCKh after translation. However, the other direction is much less successful. I believe it's mostly due to the fact that there's more info about face in COCO than in MPII. This translation models were to be then served as a baseline.

### Direction 2.






