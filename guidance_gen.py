from guidance import models, select, with_temperature


def run():
    model = "gpt2"
    lm = models.Transformers(model)
    # lm = models.LlamaCpp("models/qwen2-0_5b-instruct-q8_0.gguf") 

    print(lm + 'I like the color ' + select(['red', 'blue', 'green']))
    print(lm + 'I like the color ' + with_temperature(select(['red', 'blue', 'green']), 10))

    for i in range(10):
        result = lm + "A story: " + with_temperature(select(['He', "She"]), 1) + " put on " + select(['his', "her"]) + " " + select(['hat', "coat"])
        print(result)

if __name__ == '__main__':
    run()