import cv2
import numpy as np
import tensorflow as tf


# import the tensorflow modules and load the model
model = tf.keras.models.load_model("keras_model.h5")



# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
	#1. resize the image
    img = cv2.resize(frame,(224,224))
    #2.converting image into numpy array hand increasing to array
    test_image = np.array(img, dtype=np.float32) 
    test_image = np.expand_dims(test_image, axis=0)
    #3. normalizing the image
    normalized_image = test_image/255.0
    #4. predict result
    prediction = model.predict(normalized_image)
    print("prediction: ", prediction)

    # Display the resulting frame
    cv2.imshow('result', frame)
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
