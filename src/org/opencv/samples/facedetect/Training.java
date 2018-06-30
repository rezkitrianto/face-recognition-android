package org.opencv.samples.facedetect;

import java.util.ArrayList;
import java.util.List;

import org.opencv.samples.facedetect.FdActivity;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import android.app.Activity;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import org.opencv.android.Utils;

public class Training extends Activity {

    private static final String    TAG                 = "OCVSample::Activity";
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;
    
    public static int NUMBER_OF_CLASS = 40;
    public static int NUMBER_OF_DATASET = 10;
    public static int NUMBER_OF_TRAINING = 3;
    public static int NUMBER_OF_TESTING = NUMBER_OF_DATASET - NUMBER_OF_TRAINING;
    public static int NUMBER_OF_ALL_TRAINING = NUMBER_OF_CLASS*NUMBER_OF_TRAINING;
    public static double REJECTION_TRESHOLD = 0.4;
    public static double MIN_VALUE_MATRIX_NORM = 10;
    public static double MAX_VALUE_MATRIX_NORM = 240;
    int totalImage = NUMBER_OF_CLASS * NUMBER_OF_TRAINING;
    
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        
        setContentView(R.layout.training);
        
        Mat[] pcTrain = new Mat[totalImage];
        pcTrain = this.doTraining();
        
        Intent recogFace = new Intent(this, FdActivity.class);
        recogFace.putExtra("pcTrain", pcTrain);
		startActivity(recogFace);
    }

    @Override
    public void onPause()
    {
        super.onPause();
//        if (mOpenCvCameraView != null)
//            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        //mOpenCvCameraView.disableView();
    }
 
    private Mat[] doTraining(){
		Mat meanImage = new Mat();
		Mat[] trainImage = new Mat[totalImage];
		Mat[] pcTrain = new Mat[totalImage];
    	
    	int k = 0;
		for(int i = 1; i < NUMBER_OF_CLASS; i++){
    		for(int j = 1; j < NUMBER_OF_TRAINING; j++){
    			String drawableName = "c"+i+"_"+j;
    			Resources r = getResources();
    			int drawableId = r.getIdentifier(drawableName, "drawable", "org.opencv.samples.facedetect");
    			Bitmap bm = BitmapFactory.decodeResource(getResources(),drawableId);
    			
				Utils.bitmapToMat(bm, trainImage[k]); //THIS SHOULD BE A 3D MATRIX
				Core.add(meanImage, trainImage[k], meanImage);
				k++;
    		}
    	}
		
		Scalar totalImages = new Scalar(totalImage);
		Core.divide(meanImage, totalImages, meanImage);
		
		Mat[] substractedImg = new Mat[totalImage];
		Mat transposeSubtracted = new Mat(), multipliedMatrix = new Mat(), covMatrix = new Mat();
		k = 0;
		for(int i = 1; i < NUMBER_OF_CLASS; i++){
    		for(int j = 1; j < NUMBER_OF_TRAINING; j++){
				Core.subtract(trainImage[k], meanImage, substractedImg[k]);
				Core.transpose(substractedImg[k], transposeSubtracted);
				Core.multiply(substractedImg[k], transposeSubtracted, multipliedMatrix);
				Core.add(covMatrix, multipliedMatrix, covMatrix);
				k++;
    		}
    	}
		
		Core.divide(covMatrix, totalImages, covMatrix);

		Mat eigenValue = new Mat(), eigenVector = new Mat();
		Boolean computeEigenVector = true;
		Boolean resultEigen = Core.eigen(covMatrix, computeEigenVector, eigenValue, eigenVector);
		if(resultEigen){
			Mat diagMatrix = Mat.diag(eigenValue);
			Mat sortedDiagMatrix = new Mat();
			Core.sortIdx(diagMatrix, sortedDiagMatrix, 0);
			
			int divK = 50;
			Mat vSorted = new Mat();
			//sorting eigenvector by sorted eigenvalue
			List<Mat> listMaps = new ArrayList<Mat>();
			for(int i = 0; i < divK; i++){
				Mat rowEigenVector = eigenVector.row(i);
				listMaps.add(rowEigenVector);
			}
			Core.hconcat(listMaps, vSorted);
			
			k = 0;
			for(int i = 0; i < NUMBER_OF_CLASS*NUMBER_OF_TRAINING; i++){
				Core.multiply(trainImage[k], vSorted, pcTrain[k]);
				k++;
			}
		}

		return pcTrain;
    }
    
}
