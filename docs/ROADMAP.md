# AI 논문 학습 로드맵 (순서형)

아래 순서대로 읽으면 LLM 전체 흐름을 단계적으로 이해할 수 있습니다.

## 0) 기본기
1. [00. 선형대수+확률](posts/llm/Linear_Algebra_Foundations_concept.html)
2. [00-1. LLM 수식/기호/임베딩](posts/llm/LLM_Foundations_Notation_concept.html)
3. [00-2. Word2Vec](posts/llm/Word2Vec_concept.html)
4. [00-3. Backpropagation (역전파)](posts/foundations/Backpropagation_concept.html)
5. [00-4. 왜 Backprop이 효율적인가 — 계산 복잡도와 Gradient 구조](posts/foundations/Backprop_Efficiency_concept.html)

## 0-1) 최적화 기초 (Optimizer)
1. [Optimizer 전체 흐름 — SGD에서 AdamW까지](posts/foundations/Optimizers_concept.html)
2. [SGD 상세](posts/foundations/SGD_concept.html)
3. [Learning Rate Schedule](posts/foundations/LR_Schedule_concept.html)

## 빠른 연구자 루트
1. [Word2Vec](posts/llm/Word2Vec_concept.html)
2. [Attention Is All You Need](posts/llm/Attention_Is_All_You_Need_concept.html)
3. [왜 Transformer는 Scaling에 유리한가](posts/llm/Transformer_Scaling_concept.html)
4. Transformer 코드 읽기 (로컬 구현/실험 코드)
5. [01. BERT](posts/llm/BERT_concept.html)

## 1) 기초 아키텍처
1. [01. BERT](posts/llm/BERT_concept.html)
2. [02. RoBERTa](posts/llm/RoBERTa_concept.html)
3. [03. T5](posts/llm/T5_concept.html)

## 2) 스케일링 법칙
1. [왜 Transformer는 Scaling에 유리한가](posts/llm/Transformer_Scaling_concept.html) ← 구조적 근거
2. [04. GPT-3](posts/llm/GPT3_concept.html)
3. [05. Scaling Laws](posts/llm/Scaling_Laws_concept.html)
4. [06. Chinchilla](posts/llm/Chinchilla_concept.html)

## 3) 정렬/포스트트레이닝
1. [07. InstructGPT](posts/llm/InstructGPT_concept.html)
2. [08. Constitutional AI](posts/llm/Constitutional_AI_concept.html)
3. [09. DPO](posts/llm/DPO_concept.html)

## 4) 효율화/서빙
1. [10. Switch Transformers (MoE)](posts/llm/Switch_Transformers_concept.html)
2. [11. Mixtral (MoE)](posts/llm/Mixtral_concept.html)

## 5) RAG/에이전트
1. [12. RAG](posts/rag_agent/RAG_concept.html)
2. [13. ReAct](posts/rag_agent/ReAct_concept.html)
3. [14. Toolformer](posts/rag_agent/Toolformer_concept.html)
4. [15. Self-RAG](posts/rag_agent/Self_RAG_concept.html)

## 6) 최근 추론 트렌드
1. [16. Chain-of-Thought](posts/llm/Chain_of_Thought_concept.html)
2. [17. Self-Consistency](posts/llm/Self_Consistency_concept.html)
3. [18. DeepSeek-R1](posts/llm/DeepSeek_R1_concept.html)

## 보조 문서
- [Attention Is All You Need 개념](posts/llm/Attention_Is_All_You_Need_concept.html)
- [BERT MLM 실험](posts/llm/BERT_MLM_experiment.html)
- [RAG Retrieval 실험](posts/rag_agent/RAG_Retrieval_FAISS_experiment.html)


## 7) 비전/탐지
1. [19. YOLO](posts/vision/YOLO_concept.html)
2. [20. DETR](posts/vision/DETR_concept.html)
3. [21. CLIP](posts/vision/CLIP_concept.html)

## 8) 생성모델 (GAN/Diffusion)
1. [22. GAN](posts/generative/GAN_concept.html)
2. [23. StyleGAN](posts/generative/StyleGAN_concept.html)
3. [24. DDPM](posts/generative/DDPM_concept.html)
