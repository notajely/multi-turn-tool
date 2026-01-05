import argparse
import asyncio
import openai

import emoclassifiers.io_utils as io_utils
import emoclassifiers.classification as classification
import emoclassifiers.aggregation as aggregation


async def run_classification_on_single_conversation(
    conversation: list[dict],
    top_level_classifiers: dict[str, classification.EmoClassifier],
    sub_classifiers: dict[str, classification.EmoClassifier],
    dependency_graph: dict,
    aggregator: aggregation.Aggregator,
) -> list[dict]:
    top_level_futures_keys = []
    top_level_futures = []
    for top_level_classifier_name, top_level_classifier in top_level_classifiers.items():
        top_level_futures.append(top_level_classifier.classify_conversation(conversation))
        top_level_futures_keys.append({
            "classifier_name": top_level_classifier_name,
        })
    top_level_raw_results = await asyncio.gather(*top_level_futures)
    top_level_results = {
        key["classifier_name"]: aggregation.AnyAggregator.aggregate(raw_result)
        for key, raw_result in zip(top_level_futures_keys, top_level_raw_results)
    }
    
    sub_futures = []
    sub_futures_keys = []
    for sub_classifier_name, sub_classifier in sub_classifiers.items():
        depends_on = dependency_graph[sub_classifier_name]
        if not any(top_level_results[dep] for dep in depends_on):
            continue
        sub_futures.append(sub_classifier.classify_conversation(conversation))
        sub_futures_keys.append({
            "classifier_name": sub_classifier_name,
        })
    sub_level_raw_results = await asyncio.gather(*sub_futures)
    sub_level_results = {
        key["classifier_name"]: aggregator.aggregate(raw_result)
        for key, raw_result in zip(sub_futures_keys, sub_level_raw_results)
    }
    return {
        "top_level": top_level_results,
        "sub_level": sub_level_results,
    }


async def run_classification(
    conversation_list: list[dict],
    top_level_classifiers: dict[str, classification.EmoClassifier],
    sub_classifiers: dict[str, classification.EmoClassifier],
    dependency_graph: dict,
    aggregator: aggregation.Aggregator,
) -> list[dict]:
    print(
        f"Running {len(conversation_list)} conversations"
        f" with {len(top_level_classifiers)} top-level classifiers"
        f" and {len(sub_classifiers)} sub-classifiers"
    )
    futures = [
        run_classification_on_single_conversation(
            conversation=conversation,
            top_level_classifiers=top_level_classifiers,
            sub_classifiers=sub_classifiers,
            dependency_graph=dependency_graph,
            aggregator=aggregator,
        )
        for conversation in conversation_list
    ]
    return await asyncio.gather(*futures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--aggregation_mode", type=str, default="any")
    args = parser.parse_args()
    conversation_list = io_utils.load_jsonl(args.input_path)
    model_wrapper = classification.ModelWrapper(
        openai_client=openai.AsyncOpenAI(),
        model="gpt-4o-mini-2024-07-18",
        max_concurrent=20,
    )
    top_level_classifiers = classification.load_classifiers(
        classifier_set="v1_top_level",
        model_wrapper=model_wrapper,
    )
    sub_classifiers = classification.load_classifiers(
        classifier_set="v1",
        model_wrapper=model_wrapper,
    )
    dependency_graph = io_utils.load_json(io_utils.get_path(
        "assets/definitions/emoclassifiers_v1_dependency.json"
    ))["dependency"]
    aggregator = aggregation.AGGREGATOR_DICT[args.aggregation_mode]
    result = asyncio.run(run_classification(
        conversation_list=conversation_list,
        top_level_classifiers=top_level_classifiers,
        sub_classifiers=sub_classifiers,
        dependency_graph=dependency_graph,
        aggregator=aggregator,
    ))
    io_utils.save_jsonl(result, args.output_path)
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
