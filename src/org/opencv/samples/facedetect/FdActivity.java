package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Debug;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import org.opencv.android.Utils;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 0, 255, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;
    int counter = 0;
    public static int DISTANCE_TYPE = 2; //normal distance
    public static int CAMERA_IDX = 0;
    public static int NUMBER_OF_CLASS = 3;
    public static int NUM_OF_SEARCH_DISTANCE = 1;
    public static int NUMBER_OF_DATASET = 100;
    public static int NUMBER_OF_TRAINING = 100;
    public static int NUMBER_OF_TESTING = NUMBER_OF_DATASET - NUMBER_OF_TRAINING;
//    public static int NUMBER_OF_TESTING = 70;
    public static int NUMBER_OF_ALL_TRAINING = NUMBER_OF_CLASS * NUMBER_OF_TRAINING;
//    public static double REJECTION_TRESHOLD = 0.4;
//    public static double REJECTION_TRESHOLD = 9.04E+12;
//    public static double REJECTION_TRESHOLD = 9.50E+11;
//    public static double REJECTION_TRESHOLD = 2.21E+05; //8.21E+05 //tdk lolos
    public static double REJECTION_TRESHOLD = 3.22E+05; //8.21E+05
    public static double MIN_VALUE_MATRIX_NORM = Double.POSITIVE_INFINITY;
    public static double MAX_VALUE_MATRIX_NORM = Double.NEGATIVE_INFINITY;
    int totalImage = NUMBER_OF_CLASS * NUMBER_OF_TRAINING;
    public static int resizedWidth = 200;
    public static int resizedHeight = 200;
    
    int showCompare = 1; // 0 : yes, 1 : no
    int hasProcessed = showCompare;
    
    Mat meanImage;
    Mat vSorted;
    Mat[] pcTrain;
//    Mat meanImage = new Mat();
//    Mat vSorted = new Mat();
//    int divK = 5;
    
    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;
    
    
    private String[] className = new String[] {"RYAN","REZKI","YOGA"};

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
    	
    	class TrainingResult {
            public Mat[] pcTrain;
            public Mat meanImage;
            public Mat vSorted;
        }
    	
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    
                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");
                    
                    try {
                	    TrainingResult res = this.doTraining();
                	    pcTrain = res.pcTrain;
                	    meanImage = res.meanImage;
                	    vSorted = res.vSorted;
                	    
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);
                        
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
        
        private void showPixel(Mat bm, String title){
			Size bmsize = bm.size();
			for(int c = 0; c < bmsize.width; c++){
				for(int d = 0; d < bmsize.height; d++){
					double[] vals = bm.get(c,d);
					Log.i ("info", "notes. px "+title+" val : " +  vals[0]);
				}
			}
        }
        
		private TrainingResult doTraining(){
			Mat[] trainImage = new Mat[totalImage];
		    Mat meanImage = new Mat();
		    Mat vSorted = new Mat();
		    
		    TrainingResult res = new TrainingResult();
		    
	    	int k = 0;
			for(int i = 1; i <= NUMBER_OF_CLASS; i++){ //loop images index from 1
	    		for(int j = 1; j <= NUMBER_OF_TRAINING; j++){ //loop images index from 1
//	    			Log.i ("info", "notes. LOOP KE : "+k);
	    			
	    			//OLD LOAD DATA
	    			/*
	    			String drawableName = "c"+i+"_"+j;
	    			Resources r = getResources();
	    			int drawableId = r.getIdentifier(drawableName, "drawable", "org.opencv.samples.facedetect");
	    			Bitmap bm = BitmapFactory.decodeResource(getResources(),drawableId);
	    			
	    			Utils.bitmapToMat(b, imgToProcess);
	    			
	    			Bitmap b = this.getResizedBitmap(bm, resizedWidth, resizedHeight);
	    			Mat resizedBm = new Mat(resizedWidth, resizedHeight, CvType.CV_32FC1);
	    			Size sz = new Size(resizedWidth,resizedHeight);
	    			*/
	    			
	    			Mat bm = new Mat();
					File path = Environment.getExternalStoragePublicDirectory("Dataset");
					String filename = "c" + i + "_" + j + ".bmp";
					File file = new File(path, filename);
					filename = file.toString();
					bm = Highgui.imread(filename);
					
//					this.showPixel(bm, "loaded image");
					
					Mat imgResized = new Mat(resizedHeight, resizedWidth, CvType.CV_32FC1);
					Mat gray = new Mat(resizedHeight, resizedWidth, CvType.CV_32FC1);
	    			
					Imgproc.resize(bm, imgResized, new Size(resizedWidth, resizedHeight));
					Imgproc.cvtColor(imgResized, gray, Imgproc.COLOR_RGB2GRAY);
					trainImage[k] = gray.clone();
//					this.showPixel(gray, "gray");
					
					double min, max;
		    		MinMaxLocResult minmaxres = Core.minMaxLoc(gray);
		    		min = minmaxres.minVal;
		    		max = minmaxres.maxVal;
		    		
		    		if(min < MIN_VALUE_MATRIX_NORM) MIN_VALUE_MATRIX_NORM = min;
		    		if(max > MAX_VALUE_MATRIX_NORM) MAX_VALUE_MATRIX_NORM = max;
					
					meanImage = Mat.zeros(resizedWidth, resizedHeight, CvType.CV_32FC1);
					
					meanImage.convertTo(meanImage, CvType.CV_32FC1);
					gray.convertTo(gray, CvType.CV_32FC1);
					Core.add(meanImage, gray, meanImage);
					
					k++;
	    		}
	    	}
			
//			this.showPixel(meanImage, "all summed up");
			Scalar totalImages = new Scalar(totalImage);
//			Scalar totalImages = new Scalar(2);
			meanImage.convertTo(meanImage, CvType.CV_32FC1);
			Core.divide(meanImage, totalImages, meanImage);
//			this.showPixel(meanImage, "meanImages");
			
			Mat[] substractedImg = new Mat[totalImage];
			Mat transposeSubtracted = new Mat(resizedWidth, resizedHeight, CvType.CV_32FC1);
			Mat multipliedMatrix = new Mat(resizedWidth, resizedHeight, CvType.CV_32FC1);
			Mat covMatrix = new Mat(resizedWidth, resizedHeight, CvType.CV_32FC1);
			Mat substacted = Mat.zeros(resizedWidth, resizedHeight, CvType.CV_32FC1);
			covMatrix = Mat.zeros(resizedWidth, resizedHeight, CvType.CV_32FC1);
			
			for(int i = 0; i < NUMBER_OF_CLASS * NUMBER_OF_TRAINING; i++){ //loop trainImage, index from 0
				trainImage[i].convertTo(trainImage[i], CvType.CV_32FC1);
				meanImage.convertTo(meanImage, CvType.CV_32FC1);
				Mat trainingImage = trainImage[i].clone();
				Core.subtract(trainImage[i], meanImage, substacted);
				substractedImg[i] = substacted.clone();
				
				Core.transpose(substacted, transposeSubtracted);
//				Core.multiply(substacted, transposeSubtracted, multipliedMatrix);
				Core.multiply(transposeSubtracted, substacted, multipliedMatrix);
				Core.add(covMatrix, multipliedMatrix, covMatrix);
				
//				this.showPixel(covMatrix, "covMatrix");
	    	}
			
			Core.divide(covMatrix, totalImages, covMatrix);
//			this.showPixel(covMatrix, "covMatrix");
			
			Mat eigenValue = new Mat(), eigenVector = new Mat();
			Boolean computeEigenVector = true;
			covMatrix.convertTo(covMatrix, CvType.CV_32FC1);
			Boolean resultEigen = Core.eigen(covMatrix, computeEigenVector, eigenValue, eigenVector);
//			this.showPixel(eigenVector, "eigenVector");
			if(resultEigen){
				Mat sortedDiagMatrix = new Mat();
				Mat sortedDiagMatrixID = new Mat();
	
				Core.sortIdx(eigenValue, sortedDiagMatrixID, Core.SORT_EVERY_COLUMN + Core.SORT_DESCENDING);
				Core.sort(eigenValue, sortedDiagMatrix, Core.SORT_EVERY_ROW + Core.SORT_ASCENDING);		
				
//				Size sortedDiagMatrixIDSize = sortedDiagMatrixID.size(); //50x1
//				Log.i ("info", "notes. sortedDiagMatrixIDSize HEIGHT : " + sortedDiagMatrixIDSize.height + " sortedDiagMatrixIDSize WIDTH : "+sortedDiagMatrixIDSize.width);
//				
//				for(int c = 0; c < sortedDiagMatrixIDSize.height; c++){
//					double[] valsortedDiagMatrixID = sortedDiagMatrixID.get(c, 0);
//					Log.i ("info", "notes. valsortedDiagMatrixSize val "+c+" : " + valsortedDiagMatrixID[0]);
//				}
//				
//				Size sortedDiagMatrixSize = sortedDiagMatrix.size(); //50x1
//				Log.i ("info", "notes. sortedDiagMatrixSize HEIGHT : " + sortedDiagMatrixSize.height + " sortedDiagMatrixSize WIDTH : "+sortedDiagMatrixSize.width);
//				
//				for(int c = 0; c < sortedDiagMatrixSize.height; c++){
//					double[] valsortedDiagMatrix = sortedDiagMatrix.get(c, 0);
//					Log.i ("info", "notes. valsortedDiagMatrixSize val "+c+" : " + valsortedDiagMatrix[0]);
//				}
				
//				Mat mat1row = eigenVector.col(0);
//				Mat rowEigenVector = new Mat(resizedWidth, resizedHeight, CvType.CV_32FC1);
//				eigenVector.col(0).copyTo(rowEigenVector.col(0));
				
//				Mat vSortedNew = new Mat(resizedHeight, divK, CvType.CV_32FC1);
//				for(int i = 0; i < divK; i++){
//					double[] vals = sortedDiagMatrixID.get(i,0);
//					int intval = (int) vals[0];
//					eigenVector.col(intval).copyTo(vSortedNew.col(i));
//				}
//				
//				Size vSortedNewsize = vSortedNew.size(); //50x1
//				Log.i ("info", "notes. vSortedNewsize HEIGHT : " + vSortedNewsize.height + " vSortedNewsize WIDTH : "+vSortedNewsize.width);
//				
//				this.showPixel(vSortedNew, "vSortedNewsize");
				
//				this.showPixel(rowEigenVector, "rowEigenVector");
//				List<Mat> listMaps = new ArrayList<Mat>();
//				for(int i = 0; i < divK; i++){
//					Mat rowEigenVector = eigenVector.col(i);
//					listMaps.add(rowEigenVector);
//				}
//				Core.hconcat(listMaps, vSorted);
				
				Mat vSortedNotNormed = new Mat();
				vSortedNotNormed = eigenVector.clone();
				this.showPixel(vSortedNotNormed, "vSortedNotNormed");
				Core.normalize(vSortedNotNormed, vSorted, 0, 1, Core.NORM_MINMAX);
//				Core.normalize(vSortedNotNormed, vSorted, 0, 255, Core.NORM_MINMAX);
//				Core.normalize(vSortedNotNormed, vSorted, 0, 255, Core.NORM_L2);
				
				pcTrain = new Mat[totalImage];
				Log.i ("info", "notes. length trainImage : "+trainImage.length);
				for(int i = 0; i < NUMBER_OF_CLASS*NUMBER_OF_TRAINING; i++){
//					Log.i ("info", "notes. loop ke : "+i);
					
					Mat multipliedPcTrain = new Mat();
					
					trainImage[i].convertTo(trainImage[i], CvType.CV_32FC1);
					vSorted.convertTo(vSorted, CvType.CV_32FC1);
					Mat normalizedImage = new Mat();
					Core.subtract(trainImage[i], meanImage, normalizedImage);
					Core.multiply(trainImage[i], vSorted, multipliedPcTrain);
//					Core.multiply(normalizedImage, vSorted, multipliedPcTrain);
					pcTrain[i] = multipliedPcTrain.clone();
//					this.showPixel(normalizedImage, "pcTrain[i]");
				}
			}//end of resultEigen
			
			res.pcTrain = pcTrain;
			res.meanImage = meanImage;
			res.vSorted = vSorted;
			return res;
	    }
        
		public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
		    int width = bm.getWidth();
		    int height = bm.getHeight();
		    float scaleWidth = ((float) newWidth) / width;
		    float scaleHeight = ((float) newHeight) / height;
		    // CREATE A MATRIX FOR THE MANIPULATION
		    Matrix matrix = new Matrix();
		    // RESIZE THE BIT MAP
		    matrix.postScale(scaleWidth, scaleHeight);
		    
		    // "RECREATE" THE NEW BITMAP
		    Bitmap resizedBitmap = Bitmap.createBitmap(
		        bm, 0, 0, width, height, matrix, false);
		    bm.recycle();
		    return resizedBitmap;
		}
        
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";
        
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mLoaderCallback))
		{
		    Log.e("TEST", "Cannot connect to OpenCV Manager");
		}
        //pcTrain = (Mat[]) getIntent().getSerializableExtra("pcTrain");			//!!!!!!!!!!!!!!!!
        
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }

        
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCameraIndex(CAMERA_IDX);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    	if(hasProcessed == 0) this.compare();
    	
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        if(CAMERA_IDX == 1){
	    	Core.flip(mRgba, mRgba, 0);
	        Core.flip(mGray, mGray, 0);
        }
        
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
//                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
//                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
            mJavaDetector.detectMultiScale(
            		mGray,  //image Matrix of the type <code>CV_8U</code> containing an image where objects are detected.
            		faces, //objects Vector of rectangles where each rectangle contains the detected object.
            		1.1, //scaleFactor Parameter specifying how much the image size is reduced at each image scale.
            		2, //minNeighbors Parameter specifying how many neighbors each candidate * rectangle should have to retain it.
            		2, //flags Parameter with the same meaning for an old cascade as in the  * function <code>cvHaarDetectObjects</code>. It is not used for a new cascade.
            		new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), //minSize Minimum possible object size
            		new Size() //maxSize Maximum possible object size
            );
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++){ //looping per faces detected
    		Rect myRect = new Rect(facesArray[i].x, facesArray[i].y, facesArray[i].width, facesArray[i].height);
        	Mat roi = new Mat(mGray, myRect);
        	Mat resizedRoi = new Mat();
        	Imgproc.resize(roi, resizedRoi, new Size(resizedWidth, resizedHeight));
    		
//        	File path = Environment.getExternalStoragePublicDirectory("Result");
//        	String filename = "test.png";
//        	File file = new File(path, filename);
//
//        	Boolean bool = null;
//        	filename = file.toString();
//        	bool = Highgui.imwrite(filename, resizedRoi);
        	
        	//this will be used for the rejection system
    		double min, max;
    		MinMaxLocResult res = Core.minMaxLoc(resizedRoi);
    		min = res.minVal;
    		max = res.maxVal;
    		
        	Mat pcTest = this.getPcTest(resizedRoi);
        	MinimumDistances resMinDist = this.getMinimumDistance(pcTrain, pcTest);
        	int minIndex = resMinDist.minIndex;
        	int isReject = this.rejectionProcess(resMinDist.distances, min, max);
//        	int isReject = 0; //HARDCODE
//        	Log.i("info", "notes. minimum index : " + minIndex);
        	
        	Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        	
        	if(isReject != 1) 
        		Core.putText(mRgba, className[minIndex], facesArray[i].br(), 5, 1, FACE_RECT_COLOR);
        	else 
        		Core.putText(mRgba, "UNKNOWN", facesArray[i].br(), 5, 1, FACE_RECT_COLOR);
        	
        	
        } //end of for
        
        return mRgba;
    }
    
    class MinimumDistances {
    	public int minIndex;
        public double distances;
        public double maxValue;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Front / Back Camera");
//        mItemFace40 = menu.add("Face size 40%");
//        mItemFace30 = menu.add("Face size 30%");
//        mItemFace20 = menu.add("Front / Back Camera");
//        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
        	this.swapCamera();
//        else if (item == mItemFace40)
//            setMinFaceSize(0.4f);
//        else if (item == mItemFace30)
//            setMinFaceSize(0.3f);
//        else if (item == mItemFace20)
//        	this.swapCamera();
//        else if (item == mItemType) {
//            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
//            item.setTitle(mDetectorName[tmpDetectorType]);
//            setDetectorType(tmpDetectorType);
//        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
    
    public void compare(){
    	hasProcessed = 1;
    	int numberOfCorrect = 0;
    	
    	
    	
    	for(int i = 1; i <= NUMBER_OF_CLASS; i++){ //loop images index from 1
    		
    		File path = Environment.getExternalStoragePublicDirectory("data");
    		String filename = "";
    		if(DISTANCE_TYPE == 2) filename = "2test_"+i+".txt";
    		else filename = "test_"+i+".txt";
        	File file = new File(path, filename);
        	try {
    			file.createNewFile();
    		} catch (IOException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
        	FileOutputStream fOut = null;
    		try {
    			fOut = new FileOutputStream(file);
    		} catch (FileNotFoundException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
            OutputStreamWriter myOutWriter = 
                                    new OutputStreamWriter(fOut);
    		
    		for(int j = NUMBER_OF_TRAINING+1; j <= NUMBER_OF_DATASET; j++){ //loop images index from 1
    			Mat bm = new Mat();
				File path2 = Environment.getExternalStoragePublicDirectory("Dataset");
				String filename2 = "c" + i + "_" + j + ".bmp";
				File file2 = new File(path2, filename2);
				filename2 = file2.toString();
				bm = Highgui.imread(filename2);
				
				Mat imgResized = new Mat(resizedHeight, resizedWidth, CvType.CV_32FC1);
				Mat gray = new Mat(resizedHeight, resizedWidth, CvType.CV_32FC1);
    		
				Imgproc.resize(bm, imgResized, new Size(resizedWidth, resizedHeight));
				Imgproc.cvtColor(imgResized, gray, Imgproc.COLOR_RGB2GRAY);
				
				Mat pcTest = this.getPcTest(gray);
				MinimumDistances resMinDist = this.getMinimumDistance(pcTrain, pcTest);
	        	int minIndex = resMinDist.minIndex;
	        	if(minIndex == (i-1)) numberOfCorrect++;
//	        	Log.i ("info", "notes. minindex compare : " + minIndex + " | i : " + i);
//	        	Log.i ("info", "notes. distances compare : " + resMinDist.distances);
//	        	Log.i ("info", "notes. maxValue compare : " + resMinDist.maxValue);
	        	
	        	try {
					myOutWriter.append(Double.toString(resMinDist.maxValue) + "\n");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
    		}
    		
    		try {
    			myOutWriter.close();
    		} catch (IOException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
            try {
    			fOut.close();
    		} catch (IOException e) {
    			// TODO Auto-generated catch block
    			e.printStackTrace();
    		}
    		
    	}
    	
    	
    	
    	
    	Log.i ("info", "notes. acc numberOfCorrect : " + numberOfCorrect);
    	double acc = this.getAccuracy(numberOfCorrect);
    	Log.i ("info", "notes. acc : " + acc + "%");
    	
    }
    
    @SuppressWarnings("unused")
	private Mat getPcTest(Mat detectedFace){
    	
    	Mat pcTest = new Mat(resizedWidth, resizedHeight, CvType.CV_32FC1); 
    	Mat normalizedDetectedFace = new Mat();
    	
//    	Size detectedFaceSize = detectedFace.size(); //50x1
//		Log.i ("info", "notes. detectedFaceSize HEIGHT : " + detectedFaceSize.height + " detectedFaceSize WIDTH : "+detectedFaceSize.width);
//		
//		Size vSortedSize = vSorted.size(); //50x1
//		Log.i ("info", "notes. vSortedSize HEIGHT : " + vSortedSize.height + " vSortedSize WIDTH : "+vSortedSize.width);
//		
//		Size meanImageSize = meanImage.size(); //50x1
//		Log.i ("info", "notes. meanImageSize HEIGHT : " + meanImageSize.height + " meanImageSize WIDTH : "+meanImageSize.width);
		
		
		detectedFace.convertTo(detectedFace, CvType.CV_32FC1);
		vSorted.convertTo(vSorted, CvType.CV_32FC1);
		meanImage.convertTo(meanImage, CvType.CV_32FC1);
		
//		this.showPixel(meanImage, "meanImage");
		
		Core.subtract(detectedFace, meanImage, normalizedDetectedFace);
//		Core.multiply(normalizedDetectedFace, vSorted, pcTest);
		Core.multiply(detectedFace, vSorted, pcTest);
		
		return pcTest;
    }
    
    @SuppressWarnings("unused")
	private MinimumDistances getMinimumDistance(Mat[] pcTrain, Mat pcTest){
    	counter++;
    	MinimumDistances result = new MinimumDistances();
    	double[] dist; 
		int minIndex = 999; //999 : undefined index 
		Mat curValue = new Mat();
		Mat curValue_2 = new Mat();
		double minValue = Double.POSITIVE_INFINITY;
		double maxValue = Double.NEGATIVE_INFINITY;
//		Log.i ("info", "notes. pcTrain Length : " + pcTrain.length);

		for(int i = 0; i < totalImage; i++){
			Core.subtract(pcTest, pcTrain[i], curValue);
			if(DISTANCE_TYPE != 2) Core.multiply(curValue, curValue, curValue);
			Scalar retVal = Core.sumElems(curValue); // the distance
			double doubVal = Math.abs(retVal.val[0]);
//			Log.i ("info", "notes. doubVal val : " + doubVal);

			if(doubVal <= minValue){
				minValue = doubVal;
				minIndex = (int) Math.floor(Math.abs(i)/NUMBER_OF_TRAINING);
			}
			
			if(doubVal >= maxValue){
				maxValue = doubVal;
			}
		}
		
//		Log.i ("info", "notes. min val : " + minValue);
//		Log.i ("info", "notes. minIndex : " + minIndex);
		result.distances = minValue;
		result.minIndex = minIndex;
		result.maxValue = maxValue;
		return result;
    }
    
//    @SuppressWarnings("unused")
//	private int rejectionProcess(double difference, double min, double max){
//    	int isReject = 0;
//    	
//    	int isRejectByTreshold = this.rejectionByTreshold(difference);
//    	if(isRejectByTreshold == 0){
//    		int isRejectByMinMax = this.rejectionByMinMax(min, max);
//    		
//    		if(isRejectByMinMax == 0) isReject = 0;
//    		else isReject = 1;
//    	}
//    	else{
//    		isReject = 1;
//    	}
//    	
//    	return isReject;
//    }
    
    private int rejectionProcess(double difference, double min, double max){
    	int isReject = this.rejectionByTreshold(difference);
//    	if(isRejectByTreshold == 0){
//    		int isRejectByMinMax = this.rejectionByMinMax(min, max);
//    		
//    		if(isRejectByMinMax == 0) isReject = 0;
//    		else isReject = 1;
//    	}
//    	else{
//    		isReject = 1;
//    	}
    	
    	return isReject;
    }
    
    @SuppressWarnings("unused")
	private int rejectionByTreshold(double difference){
    	int isReject = 0;
    	if(difference > REJECTION_TRESHOLD) isReject = 1;
    	else isReject = 0;
    	return isReject;
    }
    
    private int rejectionByMinMax(double min, double max){
    	int isReject = 0;
    	if(min < MIN_VALUE_MATRIX_NORM && max > MAX_VALUE_MATRIX_NORM) isReject = 1;
    	else isReject = 0;
    	return isReject;
    }
    
	@SuppressWarnings("unused")
	private double getAccuracy(int numberOfCorrect){
		double acc;
//		Log.i ("info", "notes. 333 numberOfCorrect : " + numberOfCorrect);
//		Log.i ("info", "notes. 444 numberOfCorrect : " + NUMBER_OF_CLASS*NUMBER_OF_TESTING);
		acc = (double) numberOfCorrect / (double) (NUMBER_OF_CLASS*NUMBER_OF_TESTING);
//		Log.i ("info", "notes. @#! numberOfCorrect : " + acc);
		return acc;
	}
	
	private void showPixel(Mat bm, String title){
		Size bmsize = bm.size();
		for(int c = 0; c < bmsize.width; c++){
			for(int d = 0; d < bmsize.height; d++){
				double[] vals = bm.get(c,d);
				Log.i ("info", "notes. px "+title+" val : " +  vals[0]);
			}
		}
    }
	
	private void swapCamera() {
		CAMERA_IDX = CAMERA_IDX^1;
	    mOpenCvCameraView.disableView();
	    mOpenCvCameraView.setCameraIndex(CAMERA_IDX);
	    mOpenCvCameraView.enableView();
	}
	
}
