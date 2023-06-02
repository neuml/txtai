"""
Questions module
"""

from ..hfpipeline import HFPipeline


class Questions(HFPipeline):
    """
    Runs extractive QA for a series of questions and contexts.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("question-answering", path, quantize, gpu, model)
     
    def __call__(self, questions, contexts, workers=0, topk=1):
        self.top_k = topk
        """
        Runs a extractive question-answering model against each question-context pair, finding the best answers.

        Args:
            questions: list of questions
            contexts: list of contexts to pull answers from
            workers: number of concurrent workers to use for processing data, defaults to None

        Returns:
            list of answers
        """
        answers = []
        for x, question in enumerate(questions):
            if question and contexts[x]:
                # Run the QA pipeline
                results = self.pipeline(question=question, context=contexts[x], num_workers=workers, topk=self.top_k)
                answerlist = []
                # This is for handling lists of answers
                if isinstance(results, list):
                    for res in results:
                        # Get answer and score
                        answer, score = res["answer"], res["score"]
                        if score >= 0.05:
                            answerlist.append(answer)
                else:  # This is for handling a single answer
                    answer, score = results["answer"], results["score"]
                    if score >= 0.05:
                        answerlist.append(answer)
                # Add answers
                if answerlist:
                    answers.append(answerlist)
                else:
                    answers.append(None)
            else:
                answers.append(None)

        return answers
