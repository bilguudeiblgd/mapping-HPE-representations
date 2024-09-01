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
In direction two, main idea, given by Miraslov, was to use existing predictor like ViTPose and train the model in multi-dataset setting. Here I tried 2 methods. 

Method 1: was to use existing ViTPose which already has different decoder heads that spit outs its respective representations. Though the config was not available so a bit of hacking was necessary. In the we were able to use those. The code for the config and detector code is [https://github.com/bilguudeiblgd/ViTPose/tree/hack_vitpose_base].

Method 2: was to use ViTPose and train it on multi dataset setting. It means to batch from dataset D1, D2, at the same time and only flow loss if the keypoint exists. Then specifically, we'll train a model that predicts 17+16 = 33 keypoints, including COCO and MPII keypoints, and given an image x from COCO, we will predict for 33 keypoints, but only flow loss for first 17 points. The config is given here. [https://github.com/bilguudeiblgd/ViTPose/tree/combined_joint_prediction]. A detail ot notice, if one is to train the model, is that it has first 17 coco on the first place and then mpii keypoints but flipped.

Next, [https://github.com/bilguudeiblgd/mapping-HPE-representations/blob/main/error_distribution.ipynb] after this experiment, we saw that errors in joints are gaussian errors, and not much different between datasets. Meaning that most datasets are predicting the same thing, and there's not much bias as the errors are gaussian. Thus we merged the joints to improve generalizability and that's where we hold most of our experiments. The training&validation config is here [https://github.com/bilguudeiblgd/ViTPose/tree/combined_joint_prediction_merged].

Some detailed logs and results are here: [https://material-cowbell-938.notion.site/Vitpose-combined-daeb32b9e0574d1f80ab5f800f31b7ab]

<hr>

### Files & data left
<pre>
| AIC/
| vitpose_base_21_combined_mpii_coco/
| --- 20240729_094618.log.json
| --- best_AP_epoch_210.pth
| vitpose_base_23_combined_mpii_coco_aic/
| --- 20240821_212733.log.json
| --- best_AP_epoch_210.pth
</pre>
<hr>

<b>AIC</b> - contains AIC dataset downloaded from baidu cloud. It includes all that was included in AIC, not only keypoints

<hr>

<b>vitpose_base_21_combined_mpii_coco</b> - folder containing model that predicts COCO and MPII (union).

<b>Model explanation</b>
The models returns 21 keypoints. It was trained on MPII and COCO.

- 0..16th keypoints are COCO keypoints
- 17th keypoint correspond to MPII 9th(head_top).
- 18th keypoint correspond to MPII 8th(upper_neck).
- 19th keypoint correspond to MPII 7th(thorax).
- 20th keypoint correspond to MPII 6th(pelvis).

<hr>

<b>vitpose_base_23_combined_mpii_coco_aic</b> - folder containing model that predicts COCO, MPII, AIC (union on joints).

<b>Model explanation</b>
The models returns 23 keypoints. It was trained on MPII and COCO.
- 0..16th keypoints are COCO keypoints
- 17th keypoint correspond to MPII 9th(head_top).
- 18th keypoint correspond to MPII 8th(upper_neck).
- 19th keypoint correspond to MPII 7th(thorax).
- 20th keypoint correspond to MPII 6th(pelvis).
- 21th keypoint correspond to AIC 12th(head_top).
- 22th keypoint correspond to AIC 13th(neck).


## Final notes & tutorials

<b>Disregard:</b>

The repo itself contains, a lot of experiments and util functions for those. It used to be more detailed but I messed up with my version control and deleted every model, this repo in the gpu cluster. Around 14 models, trained or downloaded was deleted and around 15 days of work in this repo was also deleted. Though I kept the log and highlights in the notion and also in some notebook. Thus some of the notebooks might not run and they are not supposed to be "runnable material" but more like reading. It's better to create new scripts for experiments than those as the code in there is also very messy. 

<hr>

<b>Multi-dataset training:</b>

Extending and tinkering with ViTPose code is very flexible and easy, but ofc, not trivial. Experiments are done with configs. For changing the workings of the models, are done in detectors folder. For example, look at [https://github.com/bilguudeiblgd/ViTPose/blob/combined_joint_prediction_merged/mmpose/models/detectors/top_down_combined.py]. It handles multi-dataset training.

<hr>

<b>To get COCO representation on MPII (different eval on different dataset):</b>

The code [https://github.com/bilguudeiblgd/ViTPose/tree/run_mpii_model_on_coco] is here but it was written when i was starting out so there's a lot of spaghetti code I've done there. The actual only thing to change is the change the CONFIG that you are going to run and eval. Evals are done here [https://github.com/bilguudeiblgd/ViTPose/tree/run_mpii_model_on_coco/mmpose/datasets/datasets/top_down].












