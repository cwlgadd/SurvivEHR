# Callbacks for embedding labels which are specific to the token lists used in SurvivEHR paper

import torch
import numpy as np
from typing import Dict, List, Optional
from CPRD.examples.modelling.SurvivEHR.helpers import custom_mm_outcomes, expand_batch_to_context_on_tokens, filter_batch_by_context_length
from CPRD.examples.modelling.SurvivEHR.callbacks.embedding_labels import number_of_preexisting_by_token, static
from CPRD.examples.data.map_to_reduced_names import EVENT_NAME_LONG_MAP, EVENT_NAME_SHORT_MAP, convert_event_names


def filter_batch(batch: Dict[str, torch.Tensor],
                 dm,
                 expand_on_tokens: bool = True,
                 remove_on_context_length: bool = False,
                 tokens = None
                ):

    # Optionally filter out patients with long context
    if remove_on_context_length:
        # batch = filter_batch_by_context_length(batch, 10, 255)
        batch = filter_batch_by_context_length(batch, 256, np.inf)

    # Optionally expand batch context using predefined tokens. 
    #    E.g. If one sample in the batch has tokens [1,2,3,4,5] and we pass in tokens=[2,4] then expand to 
    #         two samples [1,2] and [1,2,3,4].
    if expand_on_tokens:
        if tokens is None:
            # By default use the diagnoses
            tokens = dm.encode(custom_mm_outcomes(dm))
        batch = expand_batch_to_context_on_tokens(batch, tokens)

    return batch

def Last_observed_token(batch: Dict[str, torch.Tensor],
                        dm,
                        **kwargs
                        ) -> List[str]:
    """
    Return labels for a batch based on the last ``custom_mm_outcomes`` token seen in the context.
    """
    # Decide which tokens we want to consider in any stratification method
    outcome_tokens = dm.encode(custom_mm_outcomes(dm))

    # Expand batch on the outcome tokens. 
    batch = filter_batch(batch, dm, 
                         expand_on_tokens=True,
                         tokens=outcome_tokens
                        )

    # Convert outcome tokens to a tensor (ensure it matches dtype and device of input)
    # outcome_tokens = torch.tensor(outcome_tokens, dtype=batch["tokens"].dtype, device=batch["tokens"].device)

    labels = []
    for patient_tokens in batch["tokens"]:
        
        mask = patient_tokens != 0                 # bool tensor
        idx   = mask.nonzero(as_tuple=True)        # (batch_idx, pos_idx)
        vals  = patient_tokens[idx]                # the actual non-zero token IDs
        last_seen_token = vals[-1].item()
        last_seen_event = dm.decode([last_seen_token])
        last_seen_event = EVENT_NAME_SHORT_MAP[last_seen_event]
        
        labels.append(last_seen_event)
        
    return labels, batch

def Number_of_preexisting_comorbidities(batch: Dict[str, torch.Tensor],
                                        dm,
                                        unique_only: bool = True,
                                        max_number_of_conditions: int = 5,
                                        **kwargs
                                        ) -> List[str]:
    """
    Return labels for a batch based on the number of ``custom_mm_outcomes`` in the context.
    Optionally count only unique occurences, 
    """
    batch = filter_batch(batch, dm)
    
    # Decide which tokens we want to consider in any stratification method
    outcome_tokens = dm.encode(custom_mm_outcomes(dm))

    counts = number_of_preexisting_by_token(batch,
                                            outcome_tokens,
                                            unique_only=unique_only,
                                            max_number_of_conditions=max_number_of_conditions,
                                            **kwargs)
    labels = [f"{count:2}" if count != max_number_of_conditions else f"{count:2}+" for count in counts]
    return labels, batch

def Collection_history(batch: Dict[str, torch.Tensor],
                       dm,
                       expand_on_morbidities: bool = True,
                       remove_on_context_length: bool = False,
                       **kwargs
                       ) -> List[str]:
    """
    Assigns each patient in a batch to a disease collection label 
    based on their preexisting conditions.

    Args:
        batch (Dict[str, torch.Tensor]): Input batch of patient data.
        dm: The datamodule with encode() method for converting condition names to tokens.
        expand_on_morbidities (bool): Whether to expand batch context using morbidity tokens.
        remove_on_context_length (bool): Whether to filter batch based on context length.
        **kwargs: Additional arguments passed to number_of_preexisting_by_token().

    Returns:
        List[str]: A list of labels per patient, indicating which disease 
                   collection they belong to ("None", a single group name, or "Multiple").
        batch (Dict[str, torch.Tensor]): Potentially modified batch.
    """

    batch = filter_batch(batch, dm)

    # Define disease collections and their associated condition codes
    collections = {
        "mental health": [
            "BIPOLAR", "SCHIZOPHRENIAMM_V2", "PTSDDIAGNOSIS", 
            "EATINGDISORDERS", "ANXIETY", "DEPRESSION",            
        ],
        
        "cardiovascular": [
            "PAD", "PAD_STRICT", "AORTICANEURYSM_V2", 
            "VALVULARDISEASES_V2", "STROKE_HAEMRGIC", "STROKEUNSPECIFIED_V2",
            "ISCHAEMICSTROKE_V2", "MINFARCTION", "IHDINCLUDINGMI_OPTIMALV2", 
            "HF_V3", "AF", "HYPERTENSION", "PVD_V3"
        ],
        
        "endocrine and metabolic": [
            "ADDISONS_DISEASE", "ADDISON_DISEASE", "TYPE1DM", "TYPE2DIABETES", 
            "HYPOTHYROIDISM_DRAFT_V1", "HYPERTHYROIDISM_V2", 
            "POLYCYSTIC_OVARIAN_SYNDROME_PCOS_V2", 
            "PERNICIOUSANAEMIA", "HAEMOCHROMATOSIS_V2"
        ],
    
        "respiratory": [
            "CYSTICFIBROSIS", "BRONCHIECTASIS", "COPD", 
            "ASTHMA_PUSHASTHMA", "LKA_PUSHAsthma", "LABA_PUSH_Asthma",
            "LAMA_PUSHAsthma", "OCS_PUSHAsthma2", "ICS_PUSHAsthma", 
            "SABA_PUSHAsthma", "OSA"
        ],
    
        "autoimmune and inflammatory": [
            "SYSTEMIC_SCLEROSIS", "SJOGRENSSYNDROME", "SYSTEMIC_LUPUS_ERYTHEMATOSUS",
            "PSORIATICARTHRITIS2021", "RHEUMATOIDARTHRITIS", "FIBROMYALGIA", 
            "ATOPICECZEMA", "PSORIASIS", "PMRANDGCA", "ALLERGICRHINITISCONJ"
        ],
    
        "neurological": [
            "MS", "PARKINSONS", "EPILEPSY", "PERIPHERAL_NEUROPATHY", 
            "VISUAL_IMPAIRMENT", 
            "ALL_DEMENTIA", "Dementia"
        ],
    
        "hematological and oncological": [
            "SICKLE_CELL_DISEASE_V2", "LEUKAEMIA_PREVALENCEV2", 
            "LYMPHOMA_PREVALENCE_V2", "PLASMACELL_NEOPLASM_V2", 
            "ALLCANCER_NOHAEM_NOBCC"
        ],
    
        "gastrointestinal and hepatic": [
            "CHRONIC_LIVER_DISEASE_ALCOHOL", "OTHER_CHRONIC_LIVER_DISEASE_OPTIMAL",
            "CROHNS_DISEASE", "ULCERATIVE_COLITIS", "PREVALENT_IBS_V2",
            "NAFLD_V2", 
        ],
    
        "other": [
            "ANY_DEAFNESS_HEARING_LOSS_V2",
             "SUBSTANCEMISUSE", "ALCOHOLMISUSE_V2", 
             "AUTISM",
            "OSTEOPOROSIS", "OSTEOARTHRITIS", "GOUT",
            "DOWNSSYNDROME", "CKDSTAGE3TO5",  "MENIERESDISEASE", 
             "ENDOMETRIOSIS_ADENOMYOSIS_V2",  "HIVAIDS"
        ]
    }

    labels_collections = []
    for label, conditions in collections.items():
        # Convert condition names to tokens
        outcome_tokens = dm.encode(conditions)
        # Count preexisting conditions per patient for this collection
        counts = number_of_preexisting_by_token(batch, outcome_tokens, **kwargs)
        # Record how if a patient is member of this collection for each patient
        labels_collections.append([label if count > 0 else None for count in counts])

    labels = []
    for grouped in zip(*labels_collections):
        seen_collections = [label for label in grouped if label is not None]
        unique_seen_collections = set(seen_collections)
        # if len(unique_seen_collections) == 0:
        #     labels.append("None")
        # elif len(unique_seen_collections) == 1:
        #     labels.append(unique_seen_collections[0])
        # else:
        #     labels.append("Multiple")
        labels.append(unique_seen_collections)
        
    return labels, batch

def Type2_Diabetes_history(batch: Dict[str, torch.Tensor],
                           dm,
                           max_number_of_conditions: int = 5,
                           expand_on_morbidities: bool = True,
                           **kwargs
                           ) -> List[str]:
    """
    """
    batch = filter_batch(batch, dm)

    counts = number_of_preexisting_by_token(batch,
                                            dm.encode(["TYPE2DIABETES"]),
                                            max_number_of_conditions=max_number_of_conditions, 
                                            **kwargs)
    labels = ["True" if count > 0 else "False" for count in counts]
    return labels, batch

def CVD_history(batch: Dict[str, torch.Tensor],
                dm,
                max_number_of_conditions: int = 5,
                expand_on_morbidities: bool = True,
                **kwargs
                ) -> List[str]:
    """
    """
    batch = filter_batch(batch, dm)

    counts = number_of_preexisting_by_token(batch,
                                            dm.encode(["IHDINCLUDINGMI_OPTIMALV2", "ISCHAEMICSTROKE_V2", "MINFARCTION", "STROKEUNSPECIFIED_V2", "STROKE_HAEMRGIC"]),
                                            max_number_of_conditions=max_number_of_conditions,
                                            **kwargs)
    labels = ["True" if count > 0 else "False" for count in counts]
    return labels, batch

def Hypertension_history(batch: Dict[str, torch.Tensor],
                         dm,
                         max_number_of_conditions: int = 5,
                         expand_on_morbidities: bool = True,
                         **kwargs
                         ) -> List[str]:
    """
    """

    batch = filter_batch(batch, dm)

    counts = number_of_preexisting_by_token(batch,
                                               dm.encode(["HYPERTENSION"]),
                                              max_number_of_conditions=max_number_of_conditions, 
                                              **kwargs)
    labels = ["True" if count > 0 else "False" for count in counts]
    return labels, batch
    

def IMD(batch: Dict[str, torch.Tensor], dm, **kwargs) -> List[str]:
    batch = filter_batch(batch, dm)
    labels = static(batch, dm, "IMD")
    labels = [_label for _label in labels]
    return labels, batch

def Gender(batch: Dict[str, torch.Tensor], dm, **kwargs) -> List[str]:
    batch = filter_batch(batch, dm)
    return static(batch, dm, "SEX"), batch

def Ethnicity(batch: Dict[str, torch.Tensor], dm, **kwargs) -> List[str]:
    batch = filter_batch(batch, dm)
    return static(batch, dm, "ETHNICITY"), batch

def Birth_year(batch: Dict[str, torch.Tensor], dm, **kwargs) -> List[str]:
    batch = filter_batch(batch, dm)
    labels = static(batch, dm, "birth_year")
    labels = [_label.numpy() for _label in labels]
    return labels, batch
