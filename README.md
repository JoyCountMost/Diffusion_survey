# Responsible Diffusion: A Comprehensive Survey on Safety, Ethics, and Trust in Diffusion Models


Diffusion models (DMs) have been investigated in various domains due to their ability to generate high-quality data, thereby attracting significant attention. However, similar to traditional deep learning systems, there also exist potential threats to DMs. To provide advanced and comprehensive insights into safety, ethics, and trust in DMs, this survey comprehensively elucidates its framework, threats, and countermeasures. Each threat and its countermeasures are systematically examined and categorized to facilitate thorough analysis. Furthermore, we introduce specific examples of how DMs are used, what dangers they might bring, and ways to protect against these dangers. Finally, we discuss key lessons learned, highlight open challenges related to DM security, and outline prospective research directions in this critical field. This work aims to accelerate progress not only in the technical capabilities of generative artificial intelligence but also in the maturity and wisdom of its application.

# Taxonomy of privacy risks in DMs

<table>
  <tr>
    <th>Category</th>
    <th>Ref.</th>
    <th>Attacker&apos;s knowledge</th>
    <th>Target Models</th>
    <th>Effectiveness</th>
    <th>Limitations</th>
  </tr>

  <tr>
    <td rowspan="3"><strong>Training data extraction</strong></td>
    <td><a href="https://www.usenix.org/conference/usenixsecurity23/presentation/carlini">Extracting Training Data from Diffusion Models</a></td>
    <td>Black-box, White-box</td>
    <td>OpenAI-DDPM, DDPM</td>
    <td>Extract training samples from DMs</td>
    <td>Explainability of regenerating parts of their training datasets</td>
  </tr>
  <tr>
    <td><a href="https://openreview.net/forum?id=84n3UwkH7b">Detecting, Explaining, and Mitigating Memorization in Diffusion Models</a></td>
    <td>White-box</td>
    <td>DDIM</td>
    <td>Able to achieve an AUC of 0.999 and TPR@1%FPR of 0.988</td>
    <td>A tunable threshold is required to be determined</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/2408.15721">SIDE: Surrogate Conditional Data Extraction from Diffusion Models</a></td>
    <td>White-box</td>
    <td>DDIM</td>
    <td>Highly effective in extracting memorized data</td>
    <td>Mainly uses human-labeled data, which is scarce and limits its practical use</td>
  </tr>

  <tr>
    <td><strong>Gradient inversion</strong></td>
    <td><a href="https://arxiv.org/abs/2405.20380">Gradient Inversion of Federated Diffusion Models</a></td>
    <td>Grey-box, Black-box</td>
    <td>DDPM</td>
    <td>Reconstruct high-resolution images from the gradients</td>
    <td>Single batch size</td>
  </tr>

  <tr>
    <td rowspan="11"><strong>Membership inference</strong></td>
    <td><a href="https://arxiv.org/abs/2210.00968">Membership Inference Attacks Against Text-to-Image Generation Models</a></td>
    <td>Black-box</td>
    <td>LDM, DALL-E mini</td>
    <td>Achieve remarkable attack performance</td>
    <td>Requires auxiliary datasets</td>
  </tr>
  <tr>
    <td><a href="https://proceedings.mlr.press/v202/duan23b/duan23b.pdf">Are Diffusion Models Vulnerable to Membership Inference Attacks? </a></td>
    <td>Gray-box</td>
    <td>SD, LDM, DDPM</td>
    <td>Infer the memberships of training samples</td>
    <td>Access to the step-wise query results of DMs</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/pdf/2302.03262">Membership Inference Attacks Against Diffusion Models</a></td>
    <td>White-box, Black-box</td>
    <td>DDIM</td>
    <td>The resilience of DMs to MIA is comparable to that of GANs</td>
    <td>When training samples are small</td>
  </tr>
  <tr>
    <td><a href="https://proceedings.mlr.press/v235/tang24g.html">Membership Inference Attacks on Diffusion Models via Quantile Regression</a></td>
    <td>White-box, Black-box</td>
    <td>DDPM</td>
    <td>Quantile regression models are trained to predict reconstruction loss distribution for unseen data</td>
    <td>Direct access to the trained model</td>
  </tr>
  <tr>
    <td><a href="https://openreview.net/forum?id=rpH9FcCEV6">Efficient Membership Inference for Diffusion Models by Proximal Initialization</a></td>
    <td>Gray-box</td>
    <td>DDPM, SD</td>
    <td>Utilize groundtruth trajectory and predicted point to infer memberships</td>
    <td>Access intermediate outputs of DMs</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/2405.14800">Membership Inference on T2I Diffusion via Conditional Likelihood Discrepancy</a></td>
    <td>Gray-box</td>
    <td>SD v1-4</td>
    <td>Estimate the gap between conditional image–text likelihood and images-only likelihood</td>
    <td>Evaluations under the pretraining setting are insufficient</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/2403.08487">Model Will Tell: Training Membership Inference for Diffusion Models</a></td>
    <td>Gray-box</td>
    <td>LDM, DDPM</td>
    <td>Leverage the intrinsic generative priors within the DMs</td>
    <td>Lack sufficient theoretical evidence</td>
  </tr>
  <tr>
    <td><a href="https://dl.acm.org/doi/10.1145/3664647.3681170">Unveiling Structural Memorization: Structural MIA for T2I Diffusion</a></td>
    <td>Gray-box</td>
    <td>DDIM</td>
    <td>First investigate the structural changes during the diffusion process</td>
    <td>Limited to structure-based MIA</td>
  </tr>
  <tr>
    <td><a href="https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Generated_Distributions_Are_All_You_Need_for_Membership_Inference_Attacks_WACV_2024_paper.pdf">Generated Distributions Are All You Need for Membership Inference Attacks Against Generative Models </a></td>
    <td>Black-box</td>
    <td>DDPM, DDIM, FastDPM, VQGAN, LDM, CC-FPSE</td>
    <td>Generalize MIA against various generative models including GANs, VAEs, IFs, and DDPMs</td>
    <td>Requires auxiliary datasets</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/2405.20771">Towards Black-box Membership Inference Attack for Diffusion Models</a></td>
    <td>Black-box</td>
    <td>DDIM, Diffusion Transformer, SD, DALL-E 2</td>
    <td>Only require access to the variation API of the model</td>
    <td>Requirement for a moderate diffusion step t in the variation API</td>
  </tr>
  <tr>
    <td><a href="https://www.ndss-symposium.org/ndss-paper/black-box-membership-inference-attacks-against-fine-tuned-diffusion-models/">Black-box Membership Inference Attacks Against Fine-tuned Diffusion Models</a></td>
    <td>Black-box</td>
    <td>SD, DDIM</td>
    <td>A score-based MIA tailored for modern DMs, which operates in a black-box setting</td>
    <td>Rely on a captioning model previously fine-tuned with an auxiliary dataset</td>
  </tr>

  <tr>
    <td><strong>Property inference</strong></td>
    <td><a href="https://arxiv.org/abs/2306.05208">PriSampler: Mitigating Property Inference of Diffusion Models</a></td>
    <td>Black-box</td>
    <td>TabDDPM, DDPM, SMLD, VPSDE, VESDE</td>
    <td>DMs and their sampling methods are susceptible property inference attacks</td>
    <td>Requires one part of the training datasets as auxiliary datasets</td>
  </tr>
</table>


<br/>
<br/>

# Taxonomy of defense methods against privacy issues in DMs

<table>
  <tr>
    <th>Threats</th>
    <th>Ref.</th>
    <th>Key Methods</th>
    <th>Target Models</th>
    <th>Effectiveness</th>
  </tr>

  <!-- ================= Training data extraction ================= -->
  <tr>
    <td rowspan="7"><strong>Training data extraction</strong></td>
    <!-- [37] Carlini et al., USENIX'23 -->
    <td rowspan="3">
      <a href="https://www.usenix.org/system/files/usenixsecurity23-carlini.pdf">
        Extracting Training Data from Diffusion Models
      </a>
    </td>
    <td>DP-SGD</td>
    <td rowspan="3">OpenAI-DDPM, DDPM</td>
    <td>Cause the training to fail</td>
  </tr>
  <tr>
    <td>Deduplicating</td>
    <td>Reducing memorization revealed a stronger correlation between training data extraction and duplication rates in models trained on larger-scale datasets</td>
  </tr>
  <tr>
    <td>Auditing with Canaries</td>
    <td>When auditing less leaky models, however, canary exposures computed from a single training might underestimate the true data leakage</td>
  </tr>

  <tr>
    <!-- [38] Wen et al., ICLR'24 -->
    <td rowspan="3">
      <a href="https://openreview.net/forum?id=84n3UwkH7b">
        Detecting, Explaining, and Mitigating Memorization in Diffusion Models
      </a>
    </td>
    <td>Straightforward method to detect trigger tokens</td>
    <td rowspan="3">SD</td>
    <td>Ensure a more consistent alignment between prompts and generations</td>
  </tr>
  <tr>
    <td>Inference-time mitigation method</td>
    <td rowspan="2">Successfully mitigate the memorization effect and offer a more favorable CLIP score trade-off compared to RTA</td>
  </tr>
  <tr>
    <td>Training-time mitigation method</td>
  </tr>

  <tr>
    <!-- [53] ECCV'24 cross-attention mitigation -->
    <td>
      <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09937.pdf">
        Unveiling and Mitigating Memorization in Text-to-Image Diffusion Models through Cross Attention
      </a>
    </td>
    <td>Detection and mitigation</td>
    <td>SD v1.4, SD v2.0</td>
    <td>Adding almost no computational cost, maintaining fast training and inference, significantly reducing Similarity Scores from 0.7 to 0.25–0.3</td>
  </tr>

  <!-- ================= Membership inference ================= -->
  <tr>
    <td rowspan="9"><strong>Membership inference</strong></td>
    <!-- [42] Duan et al., ICML'23 -->
    <td rowspan="5">
      <a href="https://proceedings.mlr.press/v202/duan23b/duan23b.pdf">
        Are Diffusion Models Vulnerable to Membership Inference Attacks?
      </a>
    </td>
    <td>Cutout</td>
    <td rowspan="5">DDPM</td>
    <td rowspan="2">ASR and AUC drop to some extent</td>
  </tr>
  <tr>
    <td>RandomHorizontalFlip</td>
  </tr>
  <tr>
    <td>RandAugment</td>
    <td rowspan="3">Fail to converge</td>
  </tr>
  <tr>
    <td>DP-SGD</td>
  </tr>
  <tr>
    <td>&#8467;<sub>2</sub>-regularization</td>
  </tr>

  <tr>
    <!-- [54] DualMD / DistillMD -->
    <td rowspan="2">
      <a href="https://openreview.net/forum?id=PjIe6IesEm">
        Dual-Model Defense / DistillMD 
      </a>
    </td>
    <td>DualMD</td>
    <td rowspan="2">SD v1.5, DDPM</td>
    <td>More effective for T2I DMs</td>
  </tr>
  <tr>
    <td>DistillMD</td>
    <td>Better suited for unconditional DMs</td>
  </tr>

  <tr>
    <!-- [55] MP-LoRA / SMP-LoRA -->
    <td rowspan="2">
      <a href="https://openreview.net/pdf?id=iVpribuyjP">
        Privacy-Preserving Low-Rank Adaptation against MIA for Latent Diffusion Models 
      </a>
    </td>
    <td>MP-LoRA</td>
    <td rowspan="2">SD v1.5</td>
    <td>Reduces the ASR to near-random performance while impairing LoRA&#39;s image generation capability</td>
  </tr>
  <tr>
    <td>SMP-LoRA</td>
    <td>Effectively protects membership privacy while maintaining strong image generation quality</td>
  </tr>

  <!-- ================= Property inference ================= -->
  <tr>
    <td><strong>Property inference</strong></td>
    <td>
      <a href="https://arxiv.org/abs/2306.05208">PriSampler: Mitigating Property Inference of Diffusion Models </a>
    </td>
    <td>PriSampler</td>
    <td>TabDDPM</td>
    <td>Lead adversaries to infer property proportions that approximate the predefined values specified by model owners</td>
  </tr>
</table>




