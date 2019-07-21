# Urban Reconstruction

Implementation of the paper:

**[Shape Detection from Raw LiDAR Data with Subspace Modeling](https://ieeexplore.ieee.org/document/7557019)**

Jun Wang, Kai Xu 

Urban reconstruction is an active area of research with numerous applications. It deals with the recovery of 3D geometric models of urban scenes from various types of input data, among which LiDAR scanning is becoming increasingly popular. Despite its high accuracy, fast acquisition speed, and versatility, raw LiDAR data suffer from several major imperfections which hinder its prevalence in broader scopes. Those drawbacks include anisotropy of sampling density, contamination of noise and outliers, and missing large regions. Therefore, the commonly employed robust fitting techniques should be modified accordingly. Observing that the dominant objects in urban scenes are buildings which are composed of some primitive surfaces like planes and cylinders, the problem of local structure recovery can be modeled as substructure classification. In light of this, a novel framework is proposed to first automatically detect substructures around each point, and then combine these information to achieve the reconstruction of the whole scene. The results prove that this work is very promising.
