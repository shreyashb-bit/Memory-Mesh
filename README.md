# MemoryMesh

## Privacy-Preserving AI Memory System with Verifiable Deletion

MemoryMesh is a research and engineering project focused on building a privacy-first AI memory architecture that enables secure retrieval, machine unlearning, and cryptographic proof of deletion. The system is designed to align with emerging privacy regulations such as GDPR, the EU AI Act, and India's Digital Personal Data Protection (DPDP) Act.

The project combines four core capabilities:

* Secure In-Memory Retrieval-Augmented Generation (RAG)
* Machine Unlearning using SISA Sharding
* Immutable Audit Trails using Merkle Trees
* Compliance Dashboard and APIs for transparency

---

# Project Objectives

Traditional AI systems often retain user data indefinitely, making compliance with privacy regulations difficult. MemoryMesh addresses this challenge by:

* Storing embeddings only in memory
* Encrypting user embeddings during processing
* Supporting selective machine unlearning
* Generating cryptographic deletion proofs
* Maintaining tamper-evident audit records
* Providing transparency through compliance APIs

---

# Repository Structure

      MemoryMesh/
      │
      ├── backend/
      │   ├── rag_core/                 
      │   │   ├── rag_core.py
      │   │   ├── test_deletion.py
      │   │   └── notebooks/
      │   │       └── rag_core_demo.ipynb
      │   │
      │   ├── unlearning/               
      │   │   ├── sisa_unlearn.py
      │   │   ├── unlearn_api.proto
      │   │   └── benchmarks/
      │   │       └── benchmark_report.csv
      │   │
      │   ├── audit/                   
      │   │   ├── merkle_log.py
      │   │   ├── audit_api.py
      │   │   ├── sample_proof.json
      │   │   ├── tests/
      │   │   └── docs/
      │   │
      │   ├── api/                      
      │   │   ├── main.py
      │   │   ├── auth.py
      │   │   ├── routes/
      │   │   └── middleware/
      │   │
      │   └── shared/
      │       ├── config.py
      │       ├── logger.py
      │       └── utils.py
      │
      ├── frontend/
      │   ├── src/
      │   ├── public/
      │   └── package.json
      │
      ├── docs/
      │   ├── architecture/
      │   │   ├── system_design.md
      │   │   ├── data_flow.md
      │   │   └── threat_model.md
      │   │
      │   ├── compliance/
      │   │   └── compliance_checklist.md
      │   │
      │   └── diagrams/
      │
      ├── tests/
      │   ├── integration/
      │   └── e2e/
      │
      ├── scripts/
      │   ├── setup.sh
      │   └── start_dev.sh
      │
      ├── .github/
      │   └── workflows/
      │       ├── backend-ci.yml
      │       └── frontend-ci.yml
      │
      ├── .gitignore
      ├── README.md
      ├── requirements.txt
      └── docker-compose.yml
