# Literature Review: Pre-Publication Ambiguity Detection in Internet-Drafts

**Status:** First working draft for advisor review

**Project:** Pre-Publication Ambiguity Detection in Internet-Drafts Using Revision-Aware Machine Learning and Large Language Models

**Last updated:** 2026-08-13

## Review scope and method

Internet standards are unusual software-engineering artefacts: they are written primarily in natural language, contain normative requirements alongside formal fragments, and evolve through multiple Internet-Draft revisions before publication as RFCs. This review positions the proposed project against four bodies of work: ambiguity detection in requirements and protocol specifications; automated interpretation and formalization of RFCs; RFC-guided implementation testing and compliance checking; and empirical studies of the IETF standardization process.

The local evidence base contains 18 PDF files representing 17 distinct studies, because `2506.00714v1.pdf` and `2506.00714v2.pdf` are two versions of the same work. The newer version, titled *RFCAUDIT*, is used here. The review also includes the requirements-engineering studies and SAGE cited in the project proposal. This is a targeted narrative review rather than a completed systematic literature review. Before thesis submission, the search databases, search strings, inclusion and exclusion criteria, screening process, and reference metadata should be documented explicitly.

## 1. Ambiguity in natural-language specifications

RFCs use normative terms such as MUST, SHOULD, and MAY to communicate requirement strength [1], while RFC 8174 clarifies when capitalization gives these terms their BCP 14 meaning [2]. This vocabulary improves consistency but does not make a requirement unambiguous by itself. A clause can use MUST correctly and still leave a term undefined, omit a boundary case, conflict with another section, rely on an incorrect reference, or admit several operational interpretations.

Requirements-engineering research provides a useful conceptual foundation. Berry, Kamsties, and Krieger catalogue linguistic sources of ambiguity and distinguish ambiguity from general poor writing [3]. Yang and colleagues study nocuous ambiguity: ambiguity is important when different readers can form materially different interpretations, not merely when a sentence contains a suspicious linguistic construction [4], [5]. This distinction matters for protocol standards because the target is not stylistic simplification; it is the prevention of divergent implementations and interoperability failures.

Prior automated work spans handcrafted rules, conventional supervised learning, and contextual language models. Ezzini et al. compare multiple approaches to resolving anaphoric ambiguity in requirements [6]. Veizaga et al. detect requirement smells and generate recommendations, demonstrating the continuing value of interpretable linguistic signals [7]. Izhar et al. combine engineered information with machine learning for requirements ambiguity detection [8], while Bashir et al. investigate LLM-based ambiguity detection and explanation on industrial requirements [9]. Zhao et al.'s systematic mapping study shows that NLP support for requirements engineering is broad, but persistent limitations include dataset quality, generalization, and limited industrial validation [10].

These studies justify several elements of the proposed detector, including vague-term features, anaphora and unresolved-reference checks, classical classifiers, transformers, and LLM explanations. However, requirements datasets do not reproduce the structure of an RFC. Protocol specifications have dense cross-references, specialized terminology, tables, ABNF, diagrams, and dependencies across documents. A model that performs well on isolated software requirements cannot therefore be assumed to detect logical ambiguity in an Internet-Draft.

## 2. Protocol-specific ambiguity detection

### 2.1 SAGE and semi-automated disambiguation

SAGE is an important precursor because it uses semantic parsing and interaction with protocol authors to disambiguate specification text before code generation [11]. It demonstrates that ambiguity can be exposed through an operational task: when text cannot be converted reliably into executable protocol behaviour, clarification is required. The author-in-the-loop design also provides strong evidence that a reported issue is meaningful.

Its purpose differs from the proposed project in three ways. First, SAGE targets protocol components suitable for semantic parsing and code generation rather than screening all clauses in heterogeneous drafts. Second, its manually designed grammar and author-clarification workflow limit unattended scaling. Third, it does not provide a revision-aware benchmark comparing heuristic, classical, transformer, LLM-only, and hybrid ambiguity detectors. SAGE is therefore best treated as evidence for pre-publication disambiguation and as a methodological reference, not as a directly interchangeable benchmark.

### 2.2 RFCScope and logical ambiguity in published RFCs

RFCScope is the closest direct baseline [12]. It manually analyses 273 verified technical errata from Standards Track RFCs published over an eleven-year period. Of these, 239 are classified as logical ambiguities and 34 as other issues. The logical cases are divided into three inconsistency subtypes and four under-specification subtypes:

- I-1: direct inconsistency within or across specifications;
- I-2: indirect inconsistency within or across specifications;
- I-3: inconsistency with accepted domain knowledge;
- U-1: undefined terms;
- U-2: incomplete constraints informed by implementation feedback;
- U-3: indirect under-specification within or across specifications; and
- U-4: incorrect or missing references.

RFCScope operationalizes this taxonomy in an LLM-centred pipeline. It constructs relevant cross-document context, partitions long RFCs, applies subtype-aware analysis prompts, and uses a separate evaluator to reject unsupported findings. On 20 recent RFCs, it reports 31 previously unknown logical ambiguities across 14 documents; eight were confirmed by authors and three became verified technical errata. This is strong evidence that LLMs can detect consequential logical issues that simple lexical rules miss.

RFCScope also identifies limitations that are central to the proposed research. It prioritizes precision, so false negatives may remain. Its taxonomy is derived from observed errata and may not cover every ambiguity type. Context construction can introduce noise or select an incorrect section. Dependence on evolving proprietary models affects reproducibility, and final manual inspection remains necessary. Most importantly, RFCScope analyses already published RFCs. It does not test whether the same issue could have been detected using only the information available in an earlier Internet-Draft revision.

The proposed project should reuse RFCScope's seven-subtype vocabulary and public artefacts, while preserving an independent evaluation boundary. Verified technical errata are candidate evidence, not automatically positive ambiguity labels: RFCScope itself found that some verified errata are editorial, IANA-related, suggestions, or proposals. Each erratum must be mapped to a clause and adjudicated before it becomes a gold label.

## 3. Automated interpretation and formalization of RFCs

A second research stream converts RFC prose into structured representations. These systems do not primarily classify ambiguity, but they expose the technical challenges that an ambiguity detector must handle.

PEL-BERT combines a language model with a protocol knowledge base for entity linking in RFC text [13]. Its relevance is that protocol terms cannot always be interpreted from local wording alone. Entity and reference resolution may provide features for undefined-term detection and more reliable evidence retrieval.

PROSPER uses GPT-3.5 to extract finite-state machines from RFCs and reports that combining textual content with diagram artefacts improves state and transition extraction compared with text alone [14]. Duclos et al. investigate translating RFC protocol descriptions into definitions for the Cryptographic Protocol Shapes Analyzer [15]. RFSeek generates provenance-linked, explorable protocol diagrams and supports semantic comparison of RFC logic [16]. These works show the value of structured intermediate representations and source traceability. They also show that prose, diagrams, and formal fragments should not be treated as independent channels when the intended behaviour depends on their combination.

Long et al. extend RFC interpretation to end-to-end protocol code generation through APG [17]. Their results demonstrate substantial progress in compilation and interoperability for several protocols, but the paper also reports continuing difficulties with ambiguous content, low-level logic, and generalization to broader protocol families. Successful code generation is therefore evidence that an interpretation is executable, not proof that the original clause has only one valid interpretation.

For the proposed project, these studies motivate resolved terminology, cross-reference retrieval, evidence spans, and optional structured representations. They do not provide a labelled pre-publication ambiguity benchmark, and their task-specific measures—state extraction accuracy, diagram quality, compilation, or interoperability—cannot replace clause-level precision and recall.

## 4. RFC-guided testing and implementation compliance

A substantial group of studies uses RFC text to test implementations. Although their dependent variable is implementation correctness rather than specification ambiguity, they demonstrate the practical consequences of unclear, incomplete, or evolving requirements.

RFCcert extracts certificate-validation rules from RFCs and uses them to direct differential testing of SSL/TLS implementations [18]. CHATAFL queries LLMs for protocol message and state knowledge to guide fuzzing, improving exploration and vulnerability discovery relative to the evaluated fuzzers [19]. StatePre similarly extracts state knowledge from RFCs to refine state tracking in protocol fuzzing [24]. These systems show that natural-language specifications contain actionable operational knowledge, while also illustrating how errors in extraction can propagate into downstream tests.

EBugDec is particularly relevant to revision awareness [20]. It models requirement changes across evolving RFC documents and checks whether implementations correctly incorporate primitive and derivative rules. Its evaluation spans 178 RFCs and 12 protocol-service implementations and reports 17 previously unknown evolution-related bugs. However, EBugDec studies evolution among published RFCs and corresponding code releases; it does not align successive Internet-Draft clauses or label changes that resolved ambiguity before publication.

Zheng et al. translate RFC message requirements into a formal quasi-oracle for validating parsers [21]. Because LLM-generated specifications may contain errors, implementation results are used to refine the oracle, and findings are traced back to source text. This feedback-and-traceability design is important for evidence-grounded ambiguity verification: model output should remain linked to the exact clause and supporting context, and weak automated labels should be open to correction.

RFCAUDIT uses an indexing agent and a demand-driven detection agent to compare protocol implementations with RFC requirements [22]. Wu et al. compare changes in RFC requirements with changes in TCP/IP implementations to uncover update-related inconsistencies [23]. AICCE combines retrieval, multi-agent reasoning, and executable rules for IPv6 compliance checking [25]. These recent systems indicate rapid interest in LLM-assisted standards compliance, but several are preprints and their reported implementation-level metrics are not directly comparable with ambiguity classification.

Together, these studies support the practical motivation for early ambiguity detection: inconsistent interpretations can become functional defects, security vulnerabilities, or compliance failures. They should not, however, be used as direct evidence that a clause is ambiguous. An implementation may violate a perfectly clear requirement, and two conforming implementations may choose different behaviours that the specification intentionally permits.

## 5. The IETF process and revision history as evidence

Empirical work on the IETF explains why document history is a meaningful research object. McQuistin et al. analyse 2.4 million emails, 8,711 RFCs, and 4,512 authors, reporting growth in protocol complexity and publication time [26]. Zhang et al. provide a longitudinal analysis of participant affiliations and their relationship to engagement and publication outcomes [27]. These studies frame standardization as a long-running socio-technical process rather than a single act of document publication.

Bian and Welzl address the textual workflow directly [28]. Their TextFlow systems retrieve discussions related to Internet-Draft content and use LLMs to incorporate human feedback into document editing. The work demonstrates that issue discussions and revision context can be connected to draft text and may explain why a requirement changed. This is valuable for label adjudication: a textual difference between revisions should not be called an ambiguity fix unless the surrounding evidence supports that interpretation.

These process studies do not detect ambiguity, but they support the proposed use of revision history, working-group discussions, issues, and author clarification. They also warn against a simplistic assumption that every edit is a quality correction. Revisions may reflect new functionality, terminology changes, reference maintenance, editorial work, or broader technical redesign.

## 6. Synthesis and research gap

The literature establishes five findings. First, harmful ambiguity is semantic and context-dependent, so surface-level requirement smells are useful candidate signals but insufficient final evidence. Second, protocol specifications require domain-aware interpretation across sections, documents, diagrams, and formal fragments. Third, LLMs can recover complex protocol logic, but hallucination, context selection, model drift, and cost remain material concerns. Fourth, RFC-derived rules can reveal real implementation defects, confirming the practical importance of specification quality. Fifth, the IETF's archived drafting process contains historical evidence that is largely unused by current ambiguity detectors.

No reviewed study combines all of the following:

1. detection at the Internet-Draft stage using only contemporaneously available information;
2. clause alignment across successive draft revisions and the final RFC;
3. labels that distinguish ambiguity clarification from unrelated document change;
4. a controlled comparison of heuristic, classical ML, transformer, LLM-only, and hybrid methods;
5. family-aware and temporal evaluation that prevents near-duplicate revision leakage; and
6. joint evaluation of detection, calibration, evidence faithfulness, cost, latency, and review workload.

This gap defines the project's contribution. The novelty is not simply applying another LLM to RFC text. It is constructing a revision-aware, provenance-preserving benchmark and using it to test when revision history improves detection, when LLM verification is justified, and whether a calibrated hybrid can retain high recall with fewer LLM calls.

## 7. Implications for the benchmark design

### 7.1 Unit of analysis and labels

The primary unit should be a clause-version instance, operationalized initially as a paragraph within a numbered section. Every instance should retain document family, revision, section, text span, links to aligned earlier and later clauses, and the evidence used for its label.

Labels should separate class from confidence. Gold positives may be supported by mapped verified technical errata or explicit author clarification. Silver positives may use a strong revision pattern or RFCScope-style evidence. Uncertain cases should remain visible rather than being silently removed. Changed clauses also require a revision-reason label, such as ambiguity clarification, editorial change, new functionality, terminology change, reference correction, or other technical redesign.

Negative examples need active review. The absence of an erratum is not proof that a clause is clear, especially for a recent or lightly implemented document. Negatives should be sampled from the same document families as positives to reduce topic and style confounding.

### 7.2 Leakage-resistant splits

Random clause splitting should not be the primary evaluation because adjacent draft revisions can contain identical or near-identical clauses. The principal splits should be:

- **document-family holdout:** all revisions and the final RFC from one draft family remain in one partition;
- **temporal holdout:** training uses earlier publication periods and testing uses later periods; and
- **protocol-family holdout:** one protocol area is held out to measure transfer beyond familiar terminology and structure.

A random split may be reported only as a diagnostic demonstrating the size of leakage-induced optimism. Feature construction must also obey time: a prediction for revision `-NN` cannot use a later revision, final RFC, later erratum, or discussion that did not yet exist.

### 7.3 Baselines and fair comparison

The benchmark should include a majority or class-frequency baseline, an interpretable heuristic detector, representative classical models, a compact fine-tuned transformer, a structured LLM-only detector, and the proposed hybrid router. A reproducible RFCScope-inspired published-RFC condition should be included where the released artefacts and available models permit it.

Fair comparison requires a shared test set, identical evidence-access rules, fixed prompt and model versions, and explicit treatment of abstention. SAGE and implementation-testing tools should be discussed as related systems but should not be ranked in the same metric table unless their tasks are reimplemented as clause-level ambiguity detection.

### 7.4 Metrics and statistical analysis

Because ambiguity is likely to be rare, accuracy should not be the primary measure. The main detection metrics should include precision, recall, macro-F1, F2, PR-AUC, false-negative rate, and per-subtype performance. Reliability should be assessed through calibration error, Brier score, and risk-coverage curves when abstention is allowed.

LLM and hybrid systems also require evidence correctness, hallucinated-reference rate, latency, token or compute cost, and the proportion of clauses sent to an LLM or human reviewer. Results should use fixed split manifests, repeated runs where nondeterminism applies, bootstrap confidence intervals, paired comparisons on the same test items, and ablations for revision features, retrieval, subtype prompts, and routing.

## 8. Comparative summary

| Study | Primary task | Evidence/data | Main relevance | Boundary relative to this project |
|---|---|---|---|---|
| SAGE [11] | Semi-automated disambiguation and code generation | RFC text and author clarification | Demonstrates pre-publication, author-grounded disambiguation | Grammar- and workflow-specific; no revision-aware comparative benchmark |
| RFCScope [12] | Logical ambiguity detection | Verified errata and published RFCs | Closest taxonomy, dataset, and detector baseline | Published RFC focus; proprietary-model and context-construction limitations |
| PEL-BERT [13] | Protocol entity linking | Annotated RFC entities and protocol knowledge base | Supports terminology and reference resolution | Does not classify ambiguity |
| PROSPER [14] | FSM extraction | RFC prose and diagrams | Shows value of structured artefacts and multimodal context | Evaluates extraction, not ambiguous clauses |
| Duclos et al. [15] | RFC-to-CPSA translation | Security-protocol RFC text | Connects prose with formal analysis | Early/preprint work; task-specific output |
| RFSeek [16] | Provenance-linked RFC visualization | RFC logic and diagrams | Traceable evidence and semantic comparison | Visualization rather than benchmarked ambiguity detection |
| APG [17] | RFC-to-code generation | ICMP, IGMP, NTP, and TCP specifications | Demonstrates downstream impact of interpretable RFC text | Executability does not prove a unique interpretation |
| RFCcert [18] | RFC-directed differential testing | SSL/TLS certificate rules | Shows RFC rules can provide actionable test oracles | Detects implementation bugs, not specification ambiguity |
| CHATAFL [19] | LLM-guided protocol fuzzing | Protocol messages, states, and implementations | Shows LLM-extracted protocol knowledge can improve testing | Performance measures are coverage and vulnerabilities |
| EBugDec [20] | Evolution-related implementation bug detection | 178 RFCs and 12 implementations | Strong evidence that specification evolution matters | Evolution is across published RFCs, not draft revisions |
| Zheng et al. [21] | Traceable parser validation | RFC-derived formal quasi-oracles | Useful model for evidence traceability and oracle refinement | Validates implementations rather than draft clarity |
| RFCAUDIT [22] | RFC/code conformance auditing | Six protocol implementations | Demand-driven retrieval and semantic indexing | Recent preprint; implementation-level target |
| Wu et al. [23] | Differential checks across RFC/code updates | TCP/IP standards and OS implementations | Links evolving requirements with code inconsistencies | Does not label ambiguity or pre-publication revisions |
| StatePre [24] | State-aware protocol fuzzing | RFC state knowledge and 23 programs | Demonstrates operational value of RFC state extraction | Downstream fuzzing task |
| AICCE [25] | IPv6 compliance checking | Retrieved clauses and packet samples | RAG, multi-agent adjudication, executable checks | 2026 preprint; reported compliance metrics are not ambiguity metrics |
| McQuistin et al. [26] | Empirical IETF/RFC deployment study | Emails, RFCs, authors, deployment indicators | Establishes scale and longitudinal nature of standardization | Contextual rather than detector research |
| Zhang et al. [27] | IETF affiliation evolution | Participant and organization histories | Reinforces socio-technical nature of RFC production | Indirect relevance to ambiguity |
| Bian and Welzl [28] | IETF discussion retrieval and assisted editing | Internet-Drafts and development discussions | Connects feedback with evolving draft text | Does not create ambiguity labels or compare detectors |

## 9. Conclusion

Existing work demonstrates that ambiguity in protocol specifications is real, consequential, and technically difficult to detect. Requirements-engineering methods offer interpretable linguistic signals and supervised baselines; SAGE demonstrates author-grounded disambiguation; and RFCScope provides the strongest protocol-specific taxonomy and published-RFC detector. RFC interpretation, formalization, fuzzing, and compliance studies further show that specification wording has measurable effects on implementation quality.

The unresolved question is whether consequential ambiguity can be detected reliably before publication and whether revision history provides useful evidence without leaking future information. A benchmark built from aligned Internet-Draft families, carefully adjudicated revision outcomes, mapped errata, and fixed family/time splits would address this gap. Comparing multiple detector families under the same evaluation protocol would additionally determine when LLM reasoning improves results enough to justify its cost and instability. This combination of pre-publication scope, revision-aware evidence, leakage control, and cost-aware hybrid evaluation is the central research contribution of the proposed project.

## References

[1] S. Bradner, “Key words for use in RFCs to indicate requirement levels,” RFC 2119, 1997, doi: 10.17487/RFC2119.

[2] B. Leiba, “Ambiguity of uppercase vs lowercase in RFC 2119 key words,” RFC 8174, 2017, doi: 10.17487/RFC8174.

[3] D. M. Berry, E. Kamsties, and M. M. Krieger, “From contract drafting to software specification: Linguistic sources of ambiguity,” University of Waterloo, Technical Report, 2003.

[4] H. Yang, A. Willis, A. De Roeck, and B. Nuseibeh, “A methodology for automatic identification of nocuous ambiguity,” in *Proceedings of COLING*, 2010, pp. 1218–1226.

[5] H. Yang, A. De Roeck, V. Gervasi, A. Willis, and B. Nuseibeh, “Analysing anaphoric ambiguity in natural language requirements,” *Requirements Engineering*, vol. 16, pp. 163–189, 2011.

[6] S. Ezzini, S. Abualhaija, C. Arora, and M. Sabetzadeh, “Automated handling of anaphoric ambiguity in requirements: A multi-solution study,” in *Proceedings of ICSE*, 2022, pp. 187–199.

[7] A. Veizaga, S. Y. Shin, and L. C. Briand, “Automated smell detection and recommendation in natural language requirements,” *IEEE Transactions on Software Engineering*, vol. 50, no. 4, pp. 695–720, 2024, doi: 10.1109/TSE.2024.3361033.

[8] R. Izhar, S. N. Bhatti, and S. A. Alharthi, “Bridging precision and complexity: A novel machine learning approach for ambiguity detection in software requirements,” *IEEE Access*, vol. 13, pp. 12014–12031, 2025, doi: 10.1109/ACCESS.2025.3529943.

[9] S. Bashir et al., “Requirements ambiguity detection and explanation with LLMs: An industrial study,” in *Proceedings of ICSME*, 2025, pp. 620–631.

[10] L. Zhao et al., “Natural language processing for requirements engineering: A systematic mapping study,” *ACM Computing Surveys*, vol. 54, no. 3, pp. 1–41, 2021.

[11] J. Yen, T. Lévai, Q. Ye, X. Ren, R. Govindan, and B. Raghavan, “Semi-automated protocol disambiguation and code generation,” in *Proceedings of ACM SIGCOMM*, 2021, pp. 272–286.

[12] M. Pawagi, L. Shao, H. Lee, Y. Sun, and W. Wang, “RFCScope: Detecting logical ambiguities in Internet protocol specifications,” in *Proceedings of ASE*, 2025, pp. 1234–1246, doi: 10.1109/ASE63991.2025.00106.

[13] S. Li et al., “PEL-BERT: A joint model for protocol entity linking,” arXiv:2002.00744, 2020.

[14] P. Sharma and V. Yegneswaran, “PROSPER: Extracting protocol specifications using large language models,” in *Proceedings of HotNets*, 2023, doi: 10.1145/3626111.3628205.

[15] M. Duclos, I. A. Fernandez, K. Moore, S. Mittal, and E. Zieglar, “Utilizing large language models to translate RFC protocol specifications to CPSA definitions,” arXiv:2402.00890, 2024.

[16] N. H. Rotman, T. Ferreira, H. Peleg, M. Silberstein, and A. Silva, “RFSeek and ye shall find: A tool for summary visualization and analysis of RFCs,” arXiv:2509.10216, 2025.

[17] J. Long, J. Su, and B. Han, “LLMs unleashed: Generating protocol code from RFC specifications,” in *Proceedings of AAAI*, 2026.

[18] C. Chen, C. Tian, Z. Duan, and L. Zhao, “RFC-directed differential testing of certificate validation in SSL/TLS implementations,” in *Proceedings of ICSE*, 2018, doi: 10.1145/3180155.3180226.

[19] R. Meng, M. Mirchev, M. Böhme, and A. Roychoudhury, “Large language model guided protocol fuzzing,” in *Proceedings of NDSS*, 2024, doi: 10.14722/ndss.2024.24556.

[20] J. Chen, F. Li, Q. Chen, P. Li, L. Xu, and W. Huo, “EBugDec: Detecting inconsistency bugs caused by RFC evolution in protocol implementations,” in *Proceedings of RAID*, 2023, doi: 10.1145/3607199.3607222.

[21] M. Zheng, D. Xie, Q. Shi, C. Wang, and X. Zhang, “Validating network protocol parsers with traceable RFC document interpretation,” *Proceedings of the ACM on Software Engineering*, vol. 2, ISSTA, 2025, doi: 10.1145/3728955.

[22] M. Zheng, C. Wang, X. Liu, J. Guo, S. Feng, and X. Zhang, “RFCAUDIT: AI agent for auditing protocol implementations against RFC specifications,” arXiv:2506.00714v2, 2025.

[23] Y. Wu, X. Feng, Y. Yang, and K. Xu, “Uncovering gaps between RFC updates and TCP/IP implementations: LLM-facilitated differential checks on intermediate representations,” arXiv:2510.24408, 2025.

[24] Y. Zhang, K. Zhu, J. Peng, Y. Lu, Q. Chen, and Z. Li, “StatePre: A large language model-based state-handling method for network protocol fuzzing,” *Electronics*, vol. 14, article 1931, 2025, doi: 10.3390/electronics14101931.

[25] M. W. U. Rahman et al., “AICCE: AI driven compliance checker engine,” arXiv:2604.03330, 2026.

[26] S. McQuistin et al., “Characterising the IETF through the lens of RFC deployment,” in *Proceedings of IMC*, 2021, doi: 10.1145/3487552.3487821.

[27] Y. Zhang et al., “Two decades of IETF affiliations: Evolution and impact,” in *Proceedings of ANRW*, 2025, doi: 10.1145/3744200.3744757.

[28] J. Bian and M. Welzl, “Empowering IETF collaboration with NLP search innovations and LLM-enhanced RFC writing,” in *Proceedings of ANRW*, 2025, doi: 10.1145/3744200.3744761.

## Appendix A: Local source audit

- `2506.00714v1.pdf` is an earlier version titled *An LLM Agent for Functional Bug Detection in Network Protocols* (RFCSCAN).
- `2506.00714v2.pdf` is the later version titled *RFCAUDIT: AI Agent for Auditing Protocol Implementations Against RFC Specifications* and is the version cited in this review.
- The local collection includes peer-reviewed conference/journal papers and arXiv preprints. Claims from preprints should be identified as such in the thesis and not assigned the same evidential weight automatically.
- Final thesis references should be generated from verified BibTeX/Zotero records to confirm author spelling, venue, pagination, and publication status.
