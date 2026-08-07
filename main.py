from predict_crop import predict_crop
from predict_disease import predict_disease

while True:

    print("\n==============================")
    print("🌿 PlantifyAI")
    print("==============================")
    print("1. Crop Recommendation")
    print("2. Disease Detection")
    print("3. Exit")

    choice = input("\nEnter Choice : ")

    if choice == "1":

        try:

            print("\nCrop Recommendation\n")

            n = float(input("Nitrogen : "))
            p = float(input("Phosphorus : "))
            k = float(input("Potassium : "))
            temperature = float(input("Temperature : "))
            humidity = float(input("Humidity : "))
            ph = float(input("pH : "))
            rainfall = float(input("Rainfall : "))

            crop = predict_crop(
                n,
                p,
                k,
                temperature,
                humidity,
                ph,
                rainfall
            )

            print("\n==============================")
            print("Recommended Crop :", crop)
            print("==============================")

            input("\nPress Enter to continue...")

        except Exception as e:

            print("\nERROR:")
            print(e)
            input("\nPress Enter to continue...")

    elif choice == "2":

        try:

            image = input("\nEnter Image Path : ").strip().strip('"')

            disease = predict_disease(image)

            print("\n==============================")
            print("Predicted Disease :", disease)
            print("==============================")

            input("\nPress Enter to continue...")

        except Exception as e:

            print("\nERROR:")
            print(e)
            input("\nPress Enter to continue...")

    elif choice == "3":

        print("\nThank you for using PlantifyAI 🌿")
        break

    else:

        print("\nInvalid Choice")