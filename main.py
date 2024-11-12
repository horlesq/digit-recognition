from data_preprocessing import load_preprocess_data
from model import build_load_model
from display_image import display_image

def main():
    # Load and preprocess mnist data
    (x_train, y_train), (x_test, y_test) = load_preprocess_data()

    # Load or train model
    model = build_load_model()
    
    # Train model if untrained
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

    # Save model
    model.save('digit_prediction_mnsit_model.h5')
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc}")

    # Make predictions
    predictions = model.predict(x_test)

    # Display the first image and the prediction
    display_image(x_test, predictions)
    

if __name__ == '__main__':
    main()