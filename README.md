# Responsible Diffusion: A Comprehensive Survey on Safety, Ethics, and Trust in Diffusion Models

Diffusion models (DMs) have been investigated in various domains due to their ability to generate high-quality data, thereby attracting significant attention. However, similar to traditional deep learning systems, there also exist potential threats to DMs. To provide advanced and comprehensive insights into safety, ethics, and trust in DMs, this survey comprehensively elucidates its framework, threats, and countermeasures. Each threat and its countermeasures are systematically examined and categorized to facilitate thorough analysis. Furthermore, we introduce specific examples of how DMs are used, what dangers they might bring, and ways to protect against these dangers. Finally, we discuss key lessons learned, highlight open challenges related to DM security, and outline prospective research directions in this critical field. This work aims to accelerate progress not only in the technical capabilities of generative artificial intelligence but also in the maturity and wisdom of its application.

# Taxonomy of privacy risks in DMs

<table align="center">
  <tr>
    <th align="center" valign="middle">Category</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Attacker&apos;s knowledge</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Limitations</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <tr>
    <td align="center" valign="middle" rowspan="3"><strong>Training data extraction</strong></td>
    <td align="left" valign="middle"><a href="https://www.usenix.org/conference/usenixsecurity23/presentation/carlini">Extracting Training Data from Diffusion Models</a></td>
    <td align="center" valign="middle">Black-box, White-box</td>
    <td align="center" valign="middle">OpenAI-DDPM, DDPM</td>
    <td align="center" valign="middle">Extract training samples from DMs</td>
    <td align="center" valign="middle">Explainability of regenerating parts of their training datasets</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://openreview.net/forum?id=84n3UwkH7b">Detecting, Explaining, and Mitigating Memorization in Diffusion Models</a></td>
    <td align="center" valign="middle">White-box</td>
    <td align="center" valign="middle">DDIM</td>
    <td align="center" valign="middle">Able to achieve an AUC of 0.999 and TPR@1%FPR of 0.988</td>
    <td align="center" valign="middle">A tunable threshold is required to be determined</td>
    <td align="center" valign="middle"><a href="https://github.com/YuxinWenRick/diffusion_memorization">code</a></td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://arxiv.org/abs/2410.02467">SIDE: Surrogate Conditional Data Extraction from Diffusion Models</a></td>
    <td align="center" valign="middle">White-box</td>
    <td align="center" valign="middle">DDIM</td>
    <td align="center" valign="middle">Highly effective in extracting memorized data</td>
    <td align="center" valign="middle">Mainly uses human-labeled data, which is scarce and limits its practical use</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="center" valign="middle"><strong>Gradient inversion</strong></td>
    <td align="left" valign="middle"><a href="https://arxiv.org/abs/2405.20380">Gradient Inversion of Federated Diffusion Models</a></td>
    <td align="center" valign="middle">Grey-box, Black-box</td>
    <td align="center" valign="middle">DDPM</td>
    <td align="center" valign="middle">Reconstruct high-resolution images from the gradients</td>
    <td align="center" valign="middle">Single batch size</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="center" valign="middle" rowspan="11"><strong>Membership inference</strong></td>
    <td align="left" valign="middle"><a href="https://arxiv.org/abs/2210.00968">Membership Inference Attacks Against Text-to-Image Generation Models</a></td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">LDM, DALL-E mini</td>
    <td align="center" valign="middle">Achieve remarkable attack performance</td>
    <td align="center" valign="middle">Requires auxiliary datasets</td>
    <td align="center" valign="middle"><a href="https://www.catalyzex.com/paper/membership-inference-attacks-against-text-to/code">code</a></td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://proceedings.mlr.press/v202/duan23b/duan23b.pdf">Are Diffusion Models Vulnerable to Membership Inference Attacks? </a></td>
    <td align="center" valign="middle">Gray-box</td>
    <td align="center" valign="middle">SD, LDM, DDPM</td>
    <td align="center" valign="middle">Infer the memberships of training samples</td>
    <td align="center" valign="middle">Access to the step-wise query results of DMs</td>
    <td align="center" valign="middle"><a href="https://github.com/jinhaoduan/SecMI">code</a></td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://arxiv.org/pdf/2302.03262">Membership Inference Attacks Against Diffusion Models</a></td>
    <td align="center" valign="middle">White-box, Black-box</td>
    <td align="center" valign="middle">DDIM</td>
    <td align="center" valign="middle">The resilience of DMs to MIA is comparable to that of GANs</td>
    <td align="center" valign="middle">When training samples are small</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://proceedings.mlr.press/v235/tang24g.html">Membership Inference Attacks on Diffusion Models via Quantile Regression</a></td>
    <td align="center" valign="middle">White-box, Black-box</td>
    <td align="center" valign="middle">DDPM</td>
    <td align="center" valign="middle">Quantile regression models are trained to predict reconstruction loss distribution for unseen data</td>
    <td align="center" valign="middle">Direct access to the trained model</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://openreview.net/forum?id=rpH9FcCEV6">Efficient Membership Inference for Diffusion Models by Proximal Initialization</a></td>
    <td align="center" valign="middle">Gray-box</td>
    <td align="center" valign="middle">DDPM, SD</td>
    <td align="center" valign="middle">Utilize groundtruth trajectory and predicted point to infer memberships</td>
    <td align="center" valign="middle">Access intermediate outputs of DMs</td>
    <td align="center" valign="middle"><a href="https://github.com/kong13661/PIA">code</a></td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://arxiv.org/abs/2405.14800">Membership Inference on T2I Diffusion via Conditional Likelihood Discrepancy</a></td>
    <td align="center" valign="middle">Gray-box</td>
    <td align="center" valign="middle">SD v1-4</td>
    <td align="center" valign="middle">Estimate the gap between conditional image–text likelihood and images-only likelihood</td>
    <td align="center" valign="middle">Evaluations under the pretraining setting are insufficient</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://arxiv.org/abs/2403.08487">Model Will Tell: Training Membership Inference for Diffusion Models</a></td>
    <td align="center" valign="middle">Gray-box</td>
    <td align="center" valign="middle">LDM, DDPM</td>
    <td align="center" valign="middle">Leverage the intrinsic generative priors within the DMs</td>
    <td align="center" valign="middle">Lack sufficient theoretical evidence</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://dl.acm.org/doi/10.1145/3664647.3681170">Unveiling Structural Memorization: Structural MIA for T2I Diffusion</a></td>
    <td align="center" valign="middle">Gray-box</td>
    <td align="center" valign="middle">DDIM</td>
    <td align="center" valign="middle">First investigate the structural changes during the diffusion process</td>
    <td align="center" valign="middle">Limited to structure-based MIA</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Generated_Distributions_Are_All_You_Need_for_Membership_Inference_Attacks_WACV_2024_paper.pdf">Generated Distributions Are All You Need for Membership Inference Attacks Against Generative Models </a></td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">DDPM, DDIM, FastDPM, VQGAN, LDM, CC-FPSE</td>
    <td align="center" valign="middle">Generalize MIA against various generative models including GANs, VAEs, IFs, and DDPMs</td>
    <td align="center" valign="middle">Requires auxiliary datasets</td>
    <td align="center" valign="middle"><a href="https://github.com/minxingzhang/MIAGM">code</a></td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://arxiv.org/abs/2405.20771">Towards Black-box Membership Inference Attack for Diffusion Models</a></td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">DDIM, Diffusion Transformer, SD, DALL-E 2</td>
    <td align="center" valign="middle">Only require access to the variation API of the model</td>
    <td align="center" valign="middle">Requirement for a moderate diffusion step t in the variation API</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://www.ndss-symposium.org/ndss-paper/black-box-membership-inference-attacks-against-fine-tuned-diffusion-models/">Black-box Membership Inference Attacks Against Fine-tuned Diffusion Models</a></td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD, DDIM</td>
    <td align="center" valign="middle">A score-based MIA tailored for modern DMs, which operates in a black-box setting</td>
    <td align="center" valign="middle">Rely on a captioning model previously fine-tuned with an auxiliary dataset</td>
    <td align="center" valign="middle"><a href="https://github.com/py85252876/Reconstruction-based-Attack">code</a></td>
  </tr>

  <tr>
    <td align="center" valign="middle"><strong>Property inference</strong></td>
    <td align="left" valign="middle"><a href="https://arxiv.org/abs/2306.05208">PriSampler: Mitigating Property Inference of Diffusion Models</a></td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">TabDDPM, DDPM, SMLD, VPSDE, VESDE</td>
    <td align="center" valign="middle">DMs and their sampling methods are susceptible property inference attacks</td>
    <td align="center" valign="middle">Requires one part of the training datasets as auxiliary datasets</td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>


<br/>
<br/>

# Taxonomy of defense methods against privacy issues in DMs

<table align="center">
  <tr>
    <th align="center" valign="middle">Threats</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Key Methods</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <!-- ================= Training data extraction ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="7"><strong>Training data extraction</strong></td>
    <td align="left" valign="middle" rowspan="3">
      <a href="https://www.usenix.org/system/files/usenixsecurity23-carlini.pdf">
        Extracting Training Data from Diffusion Models
      </a>
    </td>
    <td align="center" valign="middle">DP-SGD</td>
    <td align="center" valign="middle" rowspan="3">OpenAI-DDPM, DDPM</td>
    <td align="center" valign="middle">Cause the training to fail</td>
    <td align="center" valign="middle" rowspan="3">code</td>
  </tr>
  <tr>
    <td align="center" valign="middle">Deduplicating</td>
    <td align="center" valign="middle">Reducing memorization revealed a stronger correlation between training data extraction and duplication rates in models trained on larger-scale datasets</td>
  </tr>
  <tr>
    <td align="center" valign="middle">Auditing with Canaries</td>
    <td align="center" valign="middle">When auditing less leaky models, however, canary exposures computed from a single training might underestimate the true data leakage</td>
  </tr>

  <tr>
    <td align="left" valign="middle" rowspan="3">
      <a href="https://openreview.net/forum?id=84n3UwkH7b">
        Detecting, Explaining, and Mitigating Memorization in Diffusion Models
      </a>
    </td>
    <td align="center" valign="middle">Straightforward method to detect trigger tokens</td>
    <td align="center" valign="middle" rowspan="3">SD</td>
    <td align="center" valign="middle">Ensure a more consistent alignment between prompts and generations</td>
    <td align="center" valign="middle" rowspan="3"><a href="https://github.com/YuxinWenRick/diffusion_memorization">code</a></td>
  </tr>
  <tr>
    <td align="center" valign="middle">Inference-time mitigation method</td>
    <td align="center" valign="middle" rowspan="2">Successfully mitigate the memorization effect and offer a more favorable CLIP score trade-off compared to RTA</td>
  </tr>
  <tr>
    <td align="center" valign="middle">Training-time mitigation method</td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09937.pdf">
        Unveiling and Mitigating Memorization in Text-to-Image Diffusion Models through Cross Attention
      </a>
    </td>
    <td align="center" valign="middle">Detection and mitigation</td>
    <td align="center" valign="middle">SD v1.4, SD v2.0</td>
    <td align="center" valign="middle">Adding almost no computational cost, maintaining fast training and inference, significantly reducing Similarity Scores from 0.7 to 0.25–0.3</td>
    <td align="center" valign="middle"><a href="https://github.com/renjie3/MemAttn">code</a></td>
  </tr>

  <!-- ================= Membership inference ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="9"><strong>Membership inference</strong></td>
    <td align="left" valign="middle" rowspan="5">
      <a href="https://proceedings.mlr.press/v202/duan23b/duan23b.pdf">
        Are Diffusion Models Vulnerable to Membership Inference Attacks?
      </a>
    </td>
    <td align="center" valign="middle">Cutout</td>
    <td align="center" valign="middle" rowspan="5">DDPM</td>
    <td align="center" valign="middle" rowspan="2">ASR and AUC drop to some extent</td>
    <td align="center" valign="middle" rowspan="5"><a href="https://github.com/jinhaoduan/SecMI">code</a></td>
  </tr>
  <tr>
    <td align="center" valign="middle">RandomHorizontalFlip</td>
  </tr>
  <tr>
    <td align="center" valign="middle">RandAugment</td>
    <td align="center" valign="middle" rowspan="3">Fail to converge</td>
  </tr>
  <tr>
    <td align="center" valign="middle">DP-SGD</td>
  </tr>
  <tr>
    <td align="center" valign="middle">&#8467;<sub>2</sub>-regularization</td>
  </tr>

  <tr>
    <td align="left" valign="middle" rowspan="2">
      <a href="https://openreview.net/forum?id=PjIe6IesEm">
        Dual-Model Defense: Safeguarding Diffusion Models from Membership Inference Attacks through Disjoint Data Splitting 
      </a>
    </td>
    <td align="center" valign="middle">DualMD</td>
    <td align="center" valign="middle" rowspan="2">SD v1.5, DDPM</td>
    <td align="center" valign="middle">More effective for T2I DMs</td>
    <td align="center" valign="middle" rowspan="2">code</td>
  </tr>
  <tr>
    <td align="center" valign="middle">DistillMD</td>
    <td align="center" valign="middle">Better suited for unconditional DMs</td>
  </tr>

  <tr>
    <td align="left" valign="middle" rowspan="2">
      <a href="https://openreview.net/pdf?id=iVpribuyjP">
        Privacy-Preserving Low-Rank Adaptation against MIA for Latent Diffusion Models 
      </a>
    </td>
    <td align="center" valign="middle">MP-LoRA</td>
    <td align="center" valign="middle" rowspan="2">SD v1.5</td>
    <td align="center" valign="middle">Reduces the ASR to near-random performance while impairing LoRA&#39;s image generation capability</td>
    <td align="center" valign="middle" rowspan="2"><a href="https://github.com/WilliamLUO0/StablePrivateLoRA">code</a></td>
  </tr>
  <tr>
    <td align="center" valign="middle">SMP-LoRA</td>
    <td align="center" valign="middle">Effectively protects membership privacy while maintaining strong image generation quality</td>
  </tr>

  <!-- ================= Property inference ================= -->
  <tr>
    <td align="center" valign="middle"><strong>Property inference</strong></td>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2306.05208">PriSampler: Mitigating Property Inference of Diffusion Models </a>
    </td>
    <td align="center" valign="middle">PriSampler</td>
    <td align="center" valign="middle">TabDDPM</td>
    <td align="center" valign="middle">Lead adversaries to infer property proportions that approximate the predefined values specified by model owners</td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>

<br/>
<br/>

# Taxonomy of robustness risks in DMs

<table align="center">
  <tr>
    <th align="center" valign="middle">Category</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Attacker&apos;s knowledge</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Limitations</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <!-- Adversarial attacks -->
  <tr>
    <td align="center" valign="middle" rowspan="8"><strong>Adversarial attacks</strong></td>
    <!-- [61] -->
    <td align="left" valign="middle">
      <a href="https://openaccess.thecvf.com/content/CVPR2023W/AML/papers/Zhuang_A_Pilot_Study_of_Query-Free_Adversarial_Attack_Against_Stable_Diffusion_CVPRW_2023_paper.pdf">
        A Pilot Study of Query-Free Adversarial Attack against Stable Diffusion
      </a>
    </td>
    <td align="center" valign="middle">Gray-box</td>
    <td align="center" valign="middle">SD</td>
    <td align="center" valign="middle">Enable precise directional DM editing, erasing targets while preserving other regions with minimal distortion</td>
    <td align="center" valign="middle">Access to the CLIP text encoder</td>
    <td align="center" valign="middle"><a href="https://github.com/OPTML-Group/QF-Attack">code</a></td>
  </tr>

  <!-- [62] -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://aclanthology.org/2024.findings-acl.344/">
        Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks 
      </a>
    </td>
    <td align="center" valign="middle">White-box</td>
    <td align="center" valign="middle">SD v2-1</td>
    <td align="center" valign="middle">Reveal ASR differences when bidirectionally swapping entities, showing an asymmetry in adversarial attacks</td>
    <td align="center" valign="middle">Results show attack success probabilities spanning 60% to below 5%, depending on defense configurations</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <!-- [63] -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://aclanthology.org/2024.findings-emnlp.753/">
        Adversarial Attacks on Parts of Speech: An Empirical Study in Text-to-Image Generation 
      </a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD v1.5</td>
    <td align="center" valign="middle">Critical tokens and content fusion are part-of-speech (POS)-dependent, whereas suffix transferability is category-invariant</td>
    <td align="center" valign="middle">The metrics utilized in this study to assess the attack may not fully capture the visual plausibility or semantic accuracy of images after the attack</td>
    <td align="center" valign="middle"><a href="https://github.com/shahariar-shibli/Adversarial-Attack-on-POS-Tags">code</a></td>
  </tr>

  <!-- [64] -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Li_VA3_Virtually_Assured_Amplification_Attack_on_Probabilistic_Copyright_Protection_for_CVPR_2024_paper.pdf">
        VA3: Virtually Assured Amplification Attack on Probabilistic Copyright Protection for T2I GMs 
      </a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD-v1-4</td>
    <td align="center" valign="middle">The probability of generating infringing content grows monotonically with interaction length, with each query having a minimum success probability</td>
    <td align="center" valign="middle">Development of more robust copyright protection approaches</td>
    <td align="center" valign="middle"><a href="https://github.com/South7X/VA3">code</a></td>
  </tr>
  </tr>

  <!-- [36] -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://ojs.aaai.org/index.php/AAAI/article/view/28503">
        Step Vulnerability Guided Mean Fluctuation Adversarial Attack against Conditional Diffusion Models 
      </a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">LDM</td>
    <td align="center" valign="middle">The reverse process of the DMs is susceptible to the shift of the mean noise value</td>
    <td align="center" valign="middle">Choosing vulnerable steps to attack can further improve the attacking performance is not clear</td>
    <td align="center" valign="middle"><a href="https://github.com/yuhongwei22/MFA">code</a></td>
  </tr>

  <!-- [65] -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://openreview.net/forum?id=TOWdQQgMJY">
        Discovering Failure Modes of Text-guided Diffusion Models via Adversarial Search 
      </a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD, DeepFloyd, GLIDE</td>
    <td align="center" valign="middle">Investigate discrete prompt space and high-dimensional latent space to automatically identify failure modes in image generations</td>
    <td align="center" valign="middle">Rough surrogate loss functions and vanishing gradient problem</td>
    <td align="center" valign="middle"><a href="https://sage-diffusion.github.io/">code</a></td>
  </tr>

  <!-- [66] -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2410.21471">
        AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion Models (2024)
      </a>
    </td>
    <td align="center" valign="middle">White-box</td>
    <td align="center" valign="middle">Instruct-Pix2Pix, SD v1.5</td>
    <td align="center" valign="middle">Dynamically adapt to countermeasures by reducing similarity between adversarial images and NSFW embeddings in latent space, enhancing robustness</td>
    <td align="center" valign="middle">Investigate robust defenses against adversarial image attacks, considering both text and image inputs when designing safety enhancing robustness mechanisms for generative models</td>
    <td align="center" valign="middle"><a href="https://github.com/Spinozaaa/AdvI2I">code</a></td>
  </tr>

  <!-- [67] -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2310.11868">
        To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images … For Now (ECCV ’24)
      </a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD v1.4</td>
    <td align="center" valign="middle">Leverage DMs’ inherent classification to generate adversarial prompts without extra classifiers or DMs</td>
    <td align="center" valign="middle">Only evaluation on SDv1.4</td>
    <td align="center" valign="middle"><a href="https://github.com/OPTML-Group/Diffusion-MU-Attack">code</a></td>
  </tr>
</table>

<br/>
<br/>

# Taxonomy of defense methods against robustness issues in DMs

<!-- ================= TABLE 6: Defenses against adversarial attacks ================= -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Threats</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Key Methods</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <!-- Adversarial attacks -->
  <tr>
    <td align="center" valign="middle" rowspan="3"><strong>Adversarial attacks</strong></td>
    <!-- [68] Latent Guard -->
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2404.08031">
        Latent Guard: A Safety Framework for Text-to-Image Generation 
      </a>
    </td>
    <td align="center" valign="middle">Latent Guard</td>
    <td align="center" valign="middle">SD v1.5, SDXL</td>
    <td align="center" valign="middle">Allow for test time modifications of the blacklist, without retraining needs</td>
    <td align="center" valign="middle">
      <a href="https://github.com/rt219/LatentGuard">code</a>
    </td>
  </tr>

  <!-- [69] ProTIP -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04783.pdf">
        ProTIP: Probabilistic Robustness Verification on Text-to-Image Diffusion Models against Stochastic Perturbation 
      </a>
    </td>
    <td align="center" valign="middle">ProTIP</td>
    <td align="center" valign="middle">SD v1.5, SD v1.4, SDXL-Turbo</td>
    <td align="center" valign="middle">Incorporate several sequential analysis methods to dynamically determine the sample size and thus enhance the efficiency</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <!-- [70] Embedding Sanitizer -->
  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2411.10329">
        Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding (Embedding Sanitizer)
      </a>
    </td>
    <td align="center" valign="middle">Embedding Sanitizer</td>
    <td align="center" valign="middle">SD v1.4</td>
    <td align="center" valign="middle">Not only mitigates the generation of harmful concepts but also improves interpretability and controllability</td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>

<br/>
<br/>

# Taxonomy of safety issues in DMs
<!-- TABLE 7: Backdoor & Jailbreak (formatted like your TABLE3) -->
<!-- TABLE 7 (links fixed to match the survey's references) -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Category</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Attacker&apos;s knowledge</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Limitations</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <!-- ================= Backdoor ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="5"><strong>Backdoor</strong></td>
    <td align="left" valign="middle">
      <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Chou_How_to_Backdoor_Diffusion_Models_CVPR_2023_paper.pdf">[11] How to Backdoor Diffusion Models?</a>
    </td>
    <td align="center" valign="middle">White-Box</td>
    <td align="center" valign="middle">DDPM</td>
    <td align="center" valign="middle">Generate the target image once the initial noise or the initial image contains the backdoor trigger</td>
    <td align="center" valign="middle">Only consider DDPM</td>
    <td align="center" valign="middle"><a href="https://github.com/IBM/BadDiffusion">code</a></td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_TrojDiff_Trojan_Attacks_on_Diffusion_Models_With_Diverse_Targets_CVPR_2023_paper.pdf">[12] TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets</a>
    </td>
    <td align="center" valign="middle">White-Box</td>
    <td align="center" valign="middle">DDPM, DDIM</td>
    <td align="center" valign="middle">Reveal DM vulnerabilities to training data manipulations by backdooring with novel transitions that diffuse adversarial targets into a biased Gaussian distribution</td>
    <td align="center" valign="middle">1) Only consider DDPM and DDIM; 2) Induce a higher FID than benign models</td>
    <td align="center" valign="middle"><a href="https://github.com/chenweixin107/TrojDiff">code</a></td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://dl.acm.org/doi/10.1145/3581783.3612108">[13] Text-to-Image Diffusion Models can be Easily Backdoored (BadT2I)</a>
    </td>
    <td align="center" valign="middle">White-Box</td>
    <td align="center" valign="middle">SD v1.4</td>
    <td align="center" valign="middle">Inject various backdoors into the model to achieve different pre-set goals</td>
    <td align="center" valign="middle">Without considering advanced defense methods</td>
    <td align="center" valign="middle"><a href="https://github.com/sf-zhai/BadT2I">code</a></td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/6b055b95d689b1f704d8f92191cdb788-Paper-Conference.pdf">[14] VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models</a>
    </td>
    <td align="center" valign="middle">White-Box</td>
    <td align="center" valign="middle">DDPM, LDM, NCSN</td>
    <td align="center" valign="middle">The generality of our unified backdoor attack on a variety of choices in DMs, samplers, and unconditional/conditional generations</td>
    <td align="center" valign="middle">Without covering all kinds of DMs, like Cold Diffusion and Soft Diffusion</td>
    <td align="center" valign="middle"><a href="https://github.com/IBM/villandiffusion">code</a></td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2503.17724">[71] Towards Invisible Backdoor Attack on Text-to-Image Diffusion Model</a>
    </td>
    <td align="center" valign="middle">White-Box</td>
    <td align="center" valign="middle">SD v1.4</td>
    <td align="center" valign="middle">Achieve high attack success with stronger resistance to defenses</td>
    <td align="center" valign="middle">Extra time consumption compared to previous works</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <!-- ================= Jailbreak ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="9"><strong>Jailbreak</strong></td>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2404.02928">[72] Jailbreaking Prompt Attack: A Controllable Adversarial Attack Against Diffusion Models</a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">Open sourced T2I, MidJourney</td>
    <td align="center" valign="middle">Bypass both text and image safety checkers</td>
    <td align="center" valign="middle">Concept pairs are given by ChatGPT, which needs prompting that out of the automated framework</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://dl.acm.org/doi/10.1145/3576915.3623222">[73] Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes from Text-to-Image Models</a>
    </td>
    <td align="center" valign="middle">White-box</td>
    <td align="center" valign="middle">SD, LDM, DALL·E 2-demo, DALL·E mini</td>
    <td align="center" valign="middle">Easily generate realistic hateful meme variants</td>
    <td align="center" valign="middle">Full access to the T2I model, i.e., the adversary can modify model parameters to personalize image generation</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA-Diffusion_Multimodal_Attack_on_Diffusion_Models_CVPR_2024_paper.pdf">[74] MMA-Diffusion: Multimodal Attack on Diffusion Models</a>
    </td>
    <td align="center" valign="middle">White-Box, Black-Box</td>
    <td align="center" valign="middle">SD, Midjourney, Leonardo.Ai</td>
    <td align="center" valign="middle">Bypass prompt filters and safety checkers</td>
    <td align="center" valign="middle">Ideal prompt filters and safety checkers</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2308.01757">[75] AutoDAN: Automated Jailbreak via Adversarial Prompts (T2I adaptation)</a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">DALL·E 2, DALL·E 3, Cogview3, SDXL, Tongyiwanxiang, Hunyuan</td>
    <td align="center" valign="middle">Require no specific T2I architecture, and produce highly natural adversarial prompts that maintain semantic coherence</td>
    <td align="center" valign="middle">Lack classic models, such as Midjourney and Leonardo.Ai</td>
    <td align="center" valign="middle"><a href="https://github.com/rotaryhammer/code-autodan">code</a></td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://papers.nips.cc/paper_files/paper/2024/hash/4b1c0c4c6a49a0c3b8f5f2f9d7a1e8a5-Abstract-Conference.html">[76] Col-JailBreak: Collaborative Generation and Editing for Jailbreaking T2I</a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">GPT-4, DALL·E 2</td>
    <td align="center" valign="middle">Adaptive normal safe substitution, inpainting-driven injection of unsafe content, and contrastive language–image-guided collaborative optimization</td>
    <td align="center" valign="middle">Complex prompts or multiple sensitive words require repeated edits</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2408.13896">[77] HTS-Attack: Heuristic Token Search for Jailbreaking Text-to-Image Models</a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD v1.4, SafeGen, SLD</td>
    <td align="center" valign="middle">Efficiently bypass latest defense mechanisms</td>
    <td align="center" valign="middle">Lack commercial models like DALL·E 3, Midjourney, Leonardo.Ai</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2504.20376">[78] Inception: Jailbreak the Memory Mechanism of T2I Generation Systems</a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">DALL·E 3</td>
    <td align="center" valign="middle">Recursively split unsafe words into benign chunks, ensuring no semantic loss while bypassing safety filters</td>
    <td align="center" valign="middle">Consider DALL·E 3 only</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2504.11106">[79] Token-level Constraint Boundary Search for Jailbreaking Text-to-Image Models</a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD v1.4, SLD (Medium), SafeGen, DALL·E 3</td>
    <td align="center" valign="middle">Produce semantic adversarial prompts that successfully evade multi-layered defense mechanisms</td>
    <td align="center" valign="middle">Consider DALL·E 3; without Midjourney and Leonardo.Ai</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="left" valign="middle">
      <a href="https://ieeexplore.ieee.org/document/10646702">[80] SneakyPrompt: Jailbreaking Text-to-Image Generative Models</a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">DALL·E 2</td>
    <td align="center" valign="middle">Iteratively query T2I, perturbing prompts via feedback to bypass safety filters while maintaining quality</td>
    <td align="center" valign="middle">Consider DALL·E 2; without other powerful commercial models</td>
    <td align="center" valign="middle"><a href="https://github.com/Yuchen413/text2image_safety">code</a></td>
  </tr>
</table>


<br/>
<br/>

# Taxonomy of defense methods against safety issues in DMs
<!-- ================= Defense methods (Backdoor & Jailbreak) ================= -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Threats</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Key Methods</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <!-- ================= Backdoor ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="3"><strong>Backdoor</strong></td>
    <td align="left" valign="middle">
      <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11361.pdf">T2IShield: Defending Against Backdoors on Text-to-Image Diffusion Models</a>
    </td>
    <td align="center" valign="middle">T2IShield</td>
    <td align="center" valign="middle">SD v1.4</td>
    <td align="center" valign="middle">Achieve an 88.9% detection F1 score with minimal computational overhead, along with an 86.4% localization F1 score, while successfully invalidates 99% of poisoned samples</td>
    <td align="center" valign="middle"><a href="https://github.com/Robin-WZQ/T2IShield">code</a></td>
  </tr>
  <tr>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2410.12761">SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation</a>
    </td>
    <td align="center" valign="middle">SAFREE</td>
    <td align="center" valign="middle">SD v1.4</td>
    <td align="center" valign="middle">Maintain consistent safety verification while preserving output fidelity, quality, and safety</td>
    <td align="center" valign="middle"><a href="https://github.com/jaehong31/SAFREE">code</a></td>
  </tr>
  <tr>
    <td align="left" valign="middle">
      <a href="https://ojs.aaai.org/index.php/AAAI/article/view/34941/37096">UFID: A Unified Framework for Black-box Input-level Backdoor Detection on Diffusion Models</a>
    </td>
    <td align="center" valign="middle">UFID</td>
    <td align="center" valign="middle">SD v1.4</td>
    <td align="center" valign="middle">Achieve exceptional detection accuracy while maintaining superb computational efficiency</td>
    <td align="center" valign="middle"><a href="https://github.com/GuanZihan/official_UFID">code</a></td>
  </tr>

  <!-- ================= Jailbreak ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="3"><strong>Jailbreak</strong></td>
    <td align="left" valign="middle">
      <a href="https://www.researchgate.net/publication/389786654_Sparse_Autoencoder_as_a_Zero-Shot_Classifier_for_Concept_Erasing_in_Text-to-Image_Diffusion_Models">Interpret then Deactivate (ItD): Sparse Autoencoder as a Zero-Shot Classifier for Concept Erasing in T2I Diffusion Models</a>
    </td>
    <td align="center" valign="middle">Interpret then Deactivate</td>
    <td align="center" valign="middle">SD1.4</td>
    <td align="center" valign="middle">Robustly erase target concepts without forgetting on remaining concepts</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle">
      <a href="https://github.com/Visualignment/SafetyDPO">SafetyDPO: Scalable Safety Alignment for Text-to-Image Generation</a>
    </td>
    <td align="center" valign="middle">SafetyDPO</td>
    <td align="center" valign="middle">SD v1.5, SDXL</td>
    <td align="center" valign="middle">Enable T2I models to generate outputs that are not only of high quality but also aligned with safety and ethical guidelines</td>
    <td align="center" valign="middle"><a href="https://github.com/Visualignment/SafetyDPO">code</a></td>
  </tr>
  <tr>
    <td align="left" valign="middle">
      <a href="https://openreview.net/forum?id=hgTFotBRKl">SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation</a>
    </td>
    <td align="center" valign="middle">SAFREE</td>
    <td align="center" valign="middle">SD-v1.4</td>
    <td align="center" valign="middle">Show competitive results against training-based methods</td>
    <td align="center" valign="middle"><a href="https://github.com/jaehong31/SAFREE">code</a></td>
  </tr>
</table>

<br/>
<br/>

# Taxonomy of copyright issues in DMs
<!-- === Taxonomy of privacy risks in DMs — TABLE (Model extraction / Prompt stealing / Data misuse) === -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Category</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Attacker&apos;s knowledge</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Limitations</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <!-- ========== Model extraction ========== -->
  <tr>
    <td align="center" valign="middle"><strong>Model extraction</strong></td>
    <td align="left" valign="middle">
      <a href="https://visual-ai.github.io/ice/" title="ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models">[89] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models</a>
    </td>
    <td align="center" valign="middle">White-box</td>
    <td align="center" valign="middle">SD v1.5, LDM</td>
    <td align="center" valign="middle">Yield meaningful prompts that synthesize accurate, diverse images of a target concept</td>
    <td align="center" valign="middle">Without considering commercial models like DALL·E 3, Midjourney and Leonardo.Ai</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <!-- ========== Prompt stealing ========== -->
  <tr>
    <td align="center" valign="middle"><strong>Prompt stealing</strong></td>
    <td align="left" valign="middle">
      <a href="https://arxiv.org/abs/2302.09923" title="PromptStealer: Black-box Prompt Stealing Attacks against Text-to-Image Diffusion Models">[90] PromptStealer: Black-box Prompt Stealing Attacks against Text-to-Image Diffusion Models</a>
    </td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD (v1.4, v1.5, v2.0), Midjourney, DALL·E 2</td>
    <td align="center" valign="middle">Identify the modifiers within the generated image</td>
    <td align="center" valign="middle">Require a strong assumption for the defender, i.e., white-box access to the attack model</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <!-- ========== Data misuse (placeholder row as in figure) ========== -->
  <tr>
    <td align="center" valign="middle"><strong>Data misuse</strong></td>
    <td align="center" valign="middle">-</td>
    <td align="center" valign="middle">-</td>
    <td align="center" valign="middle">-</td>
    <td align="center" valign="middle">-</td>
    <td align="center" valign="middle">-</td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>


<br/>
<br/>

#  Taxonomy of defense methods against copyright issues in DMs
<!-- Taxonomy of defense methods against privacy issues in DMs (TABLE10-style) -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Threats</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Key Methods</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <!-- ================= Model extraction ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="1"><strong>Model extraction</strong></td>
    <td align="left" valign="middle">[91] NAIVEWM, FIXEDWM</td>
    <td align="center" valign="middle">NAIVEWM, FIXEDWM</td>
    <td align="center" valign="middle">LDM</td>
    <td align="center" valign="middle">Inject the watermark into the DMs and can be verified by the pre-defined prompts</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <!-- ================= Prompt stealing ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="1"><strong>Prompt stealing</strong></td>
    <td align="left" valign="middle">[92] PromptCARE</td>
    <td align="center" valign="middle">PromptCARE</td>
    <td align="center" valign="middle">BERT, RoBERTa, Facebook OPT-1.3b</td>
    <td align="center" valign="middle">Watermark and verification schemes specifically designed for unique properties of prompts and the natural language domain</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <!-- ================= Data misuse ================= -->
  <tr>
    <td align="center" valign="middle" rowspan="3"><strong>Data misuse</strong></td>
    <td align="left" valign="middle"><a href="https://arxiv.org/abs/2501.03085">[93] TabWak</a></td>
    <td align="center" valign="middle">TabWak</td>
    <td align="center" valign="middle">Tabular DM</td>
    <td align="center" valign="middle">Control the sampling of Gaussian latents for table row synthesis through the diffusion backbone</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle">[94] End-to-end watermarking</td>
    <td align="center" valign="middle">End-to-end watermarking</td>
    <td align="center" valign="middle">LDM</td>
    <td align="center" valign="middle">Enable the embedded message in the generated image to be modified as needed without retraining or fine-tuning the LDM</td>
    <td align="center" valign="middle">code</td>
  </tr>
  <tr>
    <td align="left" valign="middle"><a href="https://arankomatsuzaki.wordpress.com/2025/01/17/ft-shield-secure-copyright-verification-for-image-generation-models-with-robust-watermarking/">[95] FT-Shield</a></td>
    <td align="center" valign="middle">FT-Shield</td>
    <td align="center" valign="middle">LDM</td>
    <td align="center" valign="middle">Watermark transfer from source images to generated content, enabling copyright verification</td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>

<br/>
<br/>

# Taxonomy of fairness issues in DMs
<!-- Taxonomy of privacy risks in DMs (TABLE11 – Fairness) -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Category</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Attacker's knowledge</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Limitations</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <tr>
    <td align="center" valign="middle"><strong>Fairness</strong></td>
    <td align="center" valign="middle">[96]</td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">NCSN, DDPM, SDEM, TabDDPM</td>
    <td align="center" valign="middle">Fairness poisoning attacks: manipulate training data distribution to compromise the integrity of downstream models</td>
    <td align="center" valign="middle">Limited generalization to other models like GANs and VAEs</td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>


<br/>
<br/>

# Taxonomy of defense methods against fairness issues in DMs 
<!-- Taxonomy of defense methods against privacy issues in DMs (TABLE12 – Fairness) -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Threats</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Key Methods</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <tr>
    <td align="center" valign="middle" rowspan="4"><strong>Fairness</strong></td>
    <td align="center" valign="middle">[97]</td>
    <td align="center" valign="middle">Fair Diffusion</td>
    <td align="center" valign="middle">SD v1.5</td>
    <td align="center" valign="middle">Use learned biases and user guidance to steer the model toward a specified fairness goal during inference</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="center" valign="middle">[98]</td>
    <td align="center" valign="middle">DFT, Adjusted DFT</td>
    <td align="center" valign="middle">Runwayml/SD-v1-5</td>
    <td align="center" valign="middle">An alignment loss that steers image generation toward target distributions using adjusted gradients to optimize on-output losses</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="center" valign="middle">[99]</td>
    <td align="center" valign="middle">DifFaiRec</td>
    <td align="center" valign="middle">Diffusion Model</td>
    <td align="center" valign="middle">Design a counterfactual module to reduce the model sensitivity to protected attributes and provide mathematical explanations</td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>

<br/>
<br/>

# Taxonomy of hallucination issues in DMs
<!-- Taxonomy of privacy risks in DMs (TABLE13 – Hallucination) -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Category</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Attacker's knowledge</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Limitations</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <tr>
    <td align="center" valign="middle" rowspan="2"><strong>Hallucination</strong></td>
    <td align="center" valign="middle">[108]</td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">SD v1-4, SD v2, SD XL</td>
    <td align="center" valign="middle">Provide human-aligned, intuitive comprehensive scoring</td>
    <td align="center" valign="middle">Struggle to effectively detect key objects in synthesized landscape images</td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="center" valign="middle">[109]</td>
    <td align="center" valign="middle">Black-box</td>
    <td align="center" valign="middle">DDPM</td>
    <td align="center" valign="middle">Hallucination is indicated by high variance in the sample’s trajectory during the final backward steps</td>
    <td align="center" valign="middle">The selection of the right timesteps is key to detect hallucinations</td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>

<br/>
<br/>

# Taxonomy of defense methods against hallucination issues in DMs
<!-- Taxonomy of defense methods against privacy issues in DMs (TABLE14 – Hallucination) -->
<table align="center">
  <tr>
    <th align="center" valign="middle">Threats</th>
    <th align="center" valign="middle">Ref.</th>
    <th align="center" valign="middle">Key Methods</th>
    <th align="center" valign="middle">Target Models</th>
    <th align="center" valign="middle">Effectiveness</th>
    <th align="center" valign="middle">Code</th>
  </tr>

  <tr>
    <td align="center" valign="middle" rowspan="3"><strong>Hallucination</strong></td>
    <td align="center" valign="middle">[110]</td>
    <td align="center" valign="middle">Local Diffusion processes</td>
    <td align="center" valign="middle">DDPM, DDIM</td>
    <td align="center" valign="middle">
      OOD estimation with two modules: a “branching” module that predicts inside and outside OOD regions,
      and a “fusion” module that combines them into a unified output
    </td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="center" valign="middle">[111]</td>
    <td align="center" valign="middle">DHEaD (Hallucination Early Detection)</td>
    <td align="center" valign="middle">SD2</td>
    <td align="center" valign="middle">
      Combine cross-attention maps with a new metric, the predicted final image, to anticipate the final result
      using information from the early phases of generation
    </td>
    <td align="center" valign="middle">code</td>
  </tr>

  <tr>
    <td align="center" valign="middle">[109]</td>
    <td align="center" valign="middle">Mode Interpolation</td>
    <td align="center" valign="middle">DDPM</td>
    <td align="center" valign="middle">
      Characterize the variance in the sample’s trajectory during the final few backward sampling steps
    </td>
    <td align="center" valign="middle">code</td>
  </tr>
</table>


