# AI usage
## Tools I used, and for which functions
I used ChatGPT for the first draft of audit(). I then used it to refine specific parts of the implementation, including leakage_suspects and timestamp parsing with pd.to_datetime(time_values, errors="coerce", format="mixed") to handle mixed timestamp formats.

In metrics.py, I used ChatGPT for help with threshold_for_precision, selecting the top-k indices in precision_at_k and generating additional test cases. For example, ChatGPT suggested using np.argsort(-s, kind="stable")[:k] to select the k highest scores in descending order while preserving the original order of tied scores.

## One place where the AI was wrong, incomplete, or not what the spec said
The initial AI-generated code did not pass all the tests. For example, after implementing audit.py, running uv run pytest revealed failures in test_audit_reports_basic_problems. This showed that the implementation did not yet match the expected behavior in all cases. I also encountered test failures while refining the validation logic for leakage_suspects. I used the failing assertions to identify where the implementation needed correction, checked the code against the specifications and revised it rather than assuming the AI-generated draft was correct.

Another example was top-k selection in precision_at_k. Using argmin or argmax would not correctly select the top k items when k can be greater than one. Sorting the scores in descending order and slicing the first k indices matches the requirement more directly.

## The function I wrote without AI, and why
I wrote most of the functions in rawaj/metrics.py myself, following the specifications. I wanted to implement the metric calculations independently so I could understand their behavior instead of relying on generated code. This included translating the metric definitions into code and considering edge cases. I used AI selectively when I needed help with a specific issue, such as choosing the correct top-k indices or reasoning about the precision threshold, but I remained responsible for the main implementation and checking that it followed the requirements.