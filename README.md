# The dorsal visual pathway represents object-centered spatial relations 
files for docnet fmri project

Code Repository for "The dorsal visual pathway represents object-centered spatial relations for object recognition"

Slightly different naming conventions were used as shorthand for the tasks and data analysis. These do not always line up with the labels used in the manuscript. Below are some of the labels present in the data and tasks, and their corresponding label in the manuscript. Files with the suffix '\_supp' refer to supplemental analyses conducted with the extended sample size.


## Task Names:
    spaceloc: The 'spatial localizer.' Refers to the object-centered part relations localizer (part relations > features)

    distloc: The 'distance localizer.' Refers to the allocentric spatial relations localizer (distance > brightness)

    depthloc: The 'depth localizer.' Refers to depth localizer (3D shape > 2D shape)

    toolloc: The 'tool localizer.' Refers to the tool localizer (parietal regions: tools > non-tools; object regions: tool + non-tool > scramble)


## Condition names:
    space: relations condition from the object-centered part relations localizer 

    feature: feature condition from the object-centered part relations localizer 

    distance: distance condition from the allocentric relations localizer 

    brightness/luminance: brightness condition from the allocentric relations localizer

    3D: 3D shape condition from the depth localizer

    2D: 2D shape condition from the depth localizer

    tools: tool condition from the tool localizer

    non_tools: non-tool condition from the tool localizer


## ROI Names:
    PPC: Posterior parietal cortex. Corresponds to posterior IPS (pIPS)

    APC: Anterior parietal cortex. Corresponds to anterior IPS (aIPS)

    LO: Lateral occipital. Corresponds to LOC

    Since ROIs were functionally defined, they are generally labeled by their anatomical region + the localizer used to identify them. 
    For example, 'rLO_toolloc' refers to the region in right LOC that was defined using the tool localizer. 

    In some data files, an additional suffix may be included indicating the condition. 
    For example, lAPC_spaceloc_3D may refer to the activation to the 3D shape condition from depth localizer within a left aIPS ROI defined using the object-centered relations localizer
