# Research notes and design decisions

Reviewed: 2026-07-20.

## 1. LLM-assisted known-item search

Ma, Wu, and Ngo describe LLM-assisted query formulation as a way to diversify
query semantics and reduce language/grammar barriers in interactive
known-item search. This supports producing two intentionally different views
of one Vietnamese query:

- a fluent English visual description for the vision-language retriever;
- short synonyms for lexical or metadata retrieval.

The implementation constrains the visual field to observable content and
rejects outputs outside a strict schema. This limits invented details from
silently entering retrieval.

Reference: [Leveraging LLMs and Generative Models for Interactive Known-Item
Video Search](https://ink.library.smu.edu.sg/sis_research/8748/).

## 2. Modular parsing and planning

LLandMark separates query parsing/planning, domain reasoning, retrieval, and
reranked synthesis into specialized stages. The current task is smaller and
does not need a multi-agent runtime, but follows the same separation of
responsibilities:

1. parse the query;
2. construct independent Dev 1/Dev 2 payloads;
3. retrieve concurrently;
4. normalize and rerank.

This makes it possible to add OCR, temporal constraints, landmarks, or a paid
LLM provider without moving endpoint code.

Reference: [LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal
Interactive Video Retrieval](https://arxiv.org/abs/2603.02888).

## 3. Reciprocal Rank Fusion

Dev 1 and Dev 2 scores may use unrelated scales, so adding raw similarities is
not meaningful without calibration. RRF uses only each source's ordering:

$$\operatorname{RRF}(d)=\sum_{s \in S_d}\frac{1}{60+\operatorname{rank}_s(d)}.$$

The implementation keeps raw scores for debugging but never mixes them into
the fused score. Duplicate frame IDs within one source contribute only once.

Reference: [Reciprocal Rank Fusion outperforms Condorcet and individual Rank
Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).

## 4. Query expansion

The paper *Exploring the Best Practices of Query Expansion with Large Language
Models* introduces **Multi-Text Generation Integration (MuGI)**. Its practical
lesson for this service is that expansion should offer multiple useful lexical
views while remaining tied to the original intent. The initial parser therefore
returns a compact, deduplicated list of synonyms instead of one unconstrained
paragraph.

Reference: [MuGI paper, Findings of EMNLP
2024](https://aclanthology.org/2024.findings-emnlp.103/).

Important bibliographic correction: **QueryGym is not the toolkit introduced
by the MuGI paper**. QueryGym is a separate reproducibility toolkit published
later for benchmarking LLM-based query reformulation.

Reference: [QueryGym paper](https://ls3.rnet.torontomu.ca/wp-content/uploads/2026/01/querygym.pdf).

## 5. NTCIR-18 Lifelog-6

The NTCIR-18 systems reinforce a staged retrieval design. LifeIR combines
CLIP-based retrieval, query rewriting, event-based candidate expansion, and
MLLM posterior filtering. Other teams report LLM-based query transformation,
pseudo-relevance feedback, and separate retrieval/reader stages. For this
Online Serving milestone, the parser and RRF layer are intentionally lightweight
and low-latency; event expansion or MLLM posterior filtering can be added later
as separate services.

References:

- [Official NTCIR-18 Lifelog-6 proceedings index](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings18/NTCIR/abstract_ntcir.html)
- [LifeIR at the NTCIR-18 Lifelog-6 Task](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings18/pdf/ntcir/04-NTCIR18-LIFELOG-ChenJ.pdf)
- [Overview of the NTCIR-18 Lifelog-6 Task](https://repository.nii.ac.jp/record/2002046/files/01-NTCIR18-OV-LIFELOG-ZhouL.pdf)

