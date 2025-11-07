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
