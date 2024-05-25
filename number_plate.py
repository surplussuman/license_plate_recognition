import cv2
import easyocr
import pandas as pd
harcascade = "model/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

cap.set(3, 640) # width
cap.set(4, 480) #height

reader = easyocr.Reader(['en'])
data = []

min_area = 500
count = 0

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y+h, x:x+w]
            

            gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            _, thresh_roi = cv2.threshold(blur_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours and get the largest one
            contours, _ = cv2.findContours(thresh_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding box coordinates of the contour
            x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(largest_contour)

            # Crop the contour region
            plate_crop = thresh_roi[y_contour:y_contour + h_contour, x_contour:x_contour + w_contour]

            '''output = reader.readtext(thresh_img)
            
            if output:
                plate_number = output[0][-2]
                data.append({"Plate Number": plate_number})
                df = pd.DataFrame(data)
                df.to_excel("plate_numbers.xlsx", index=False)
                print("Plate Number:", plate_number)'''

            cv2.imshow("ROI", plate_crop)


    
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", plate_crop)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results",img)
        cv2.waitKey(500)
 # OCR
        output = reader.readtext("plates/scaned_img_" + str(count) + ".jpg", allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if output:
            plate_number = output[0][-2]
            data.append({"Plate Number": plate_number})
            df = pd.DataFrame(data)
            df.to_excel("plate_numbers.xlsx", index=False)
            print("Plate Number:", plate_number)
                
        count += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break   

