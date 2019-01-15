from skmultiflow.data.led_generator import LEDGenerator

if __name__ == '__main__':
    stream = LEDGenerator(random_state=112, noise_percentage=0.28, has_noise=True)
    stream.prepare_for_use()
    print(stream.next_sample())