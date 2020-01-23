#include "BPNet_3L.cpp"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cfloat>
using namespace std;

const float LEARNING_RATE    = 0.01;
const float MOMENTUM_RATE    = 0.01;
const float TRAIN_TEST_RATIO = .80;
const int   MAX_ITER         = 25000;  //25000;
const int   NUM_CLASSES      = 2;
const int   NUM_HIDDEN       = 6;
const int   NUM_DATA         = 100;
const int   RAW_DATA_DIM     = 13;
const int   DATA_DIM         = 12;
const int   OUTPUT_DIM       = 2;
const int   LOOKBEHIND_AMNT  = 2000; // 10000;
const int   IS_POP_THRESH    = 75;

template<class T, int n, int m>        
    Matrix<T,n,m>            readCSV(string); 
template<class T, int n, int m, int k> 
    Matrix<T,n,m>            preprocess(Matrix<T,n,k>);
template<class T, int n, int m> 
    array<int,NUM_CLASSES+1> getClassIndices (Matrix<T,n,m> rawData);
template<class T, int n> 
    float                    rmse(array<T,n>, array<T,n>);
template<class T, int n, int m> 
    Matrix<T,n,m>            getExpectedVectors(Matrix<float,NUM_DATA, RAW_DATA_DIM>);   
template<class T, int n>
    int                      classOf(array<T,n>);
template<class T, int n>
    float                    avgOf(array<T,n> arr);
    
void displayHeader();
int classOf(int);

int main() {

//============================== Initializations ==============================
    BPNet_3L<float, DATA_DIM, NUM_HIDDEN, OUTPUT_DIM> network(LEARNING_RATE,
                                                              MOMENTUM_RATE);
    Matrix  <float, NUM_DATA,   RAW_DATA_DIM>         rawData;
    Matrix  <float, NUM_DATA,   DATA_DIM>             inputData;
    Matrix  <float, NUM_DATA,   OUTPUT_DIM>           expectedOutput;
    Matrix  <int,   OUTPUT_DIM, OUTPUT_DIM>           confusionMatrix;  
    //Matrix  <float, NUM_HIDDEN, DATA_DIM>             intialWeights_ih;
    //Matrix  <float, OUTPUT_DIM, NUM_HIDDEN>           intialWeights_ho;
    
    array<int,   LOOKBEHIND_AMNT>  historyBuffer; 
    //array<int,   NUM_CLASSES+1>    classIndices;
    array<float, OUTPUT_DIM>       outVec;
    float                          bufferAvg;
    float                          testRMSE;
    bool                           notConverged;
    int                            bufferCounter=0;

    //displayHeader();
    //intialWeights_ih = network.getWeights_ih();
    //intialWeights_ho = network.getWeights_ho();
    

    rawData      = readCSV   <float, NUM_DATA, RAW_DATA_DIM>("pop.csv");
    inputData    = preprocess<float, NUM_DATA, DATA_DIM, RAW_DATA_DIM>(rawData);
    //classIndices = getClassIndices<float,NUM_DATA,RAW_DATA_DIM>(rawData);
    expectedOutput = getExpectedVectors<float,NUM_DATA,OUTPUT_DIM>(rawData);

    int numIter = 0; 
    while((numIter < MAX_ITER && notConverged) || numIter < LOOKBEHIND_AMNT)
    {
        // --- Train the Network ---
        for (int i = 0; i < (int)(NUM_DATA * TRAIN_TEST_RATIO); i++)
        {
            network.train(inputData[i], expectedOutput[i]);
        }

        // --- Test the Network --- 
        testRMSE = 0.0;
        for (int i = (int)(NUM_DATA * TRAIN_TEST_RATIO); i < NUM_DATA; i++)
        {
                outVec = network.classify(inputData[i]);
                testRMSE += rmse<float, OUTPUT_DIM>(outVec, expectedOutput[i]);
        }
        //cout << "Iteration: " << numIter << ". ";
        //cout << "Sum RMSE: " << testRMSE << "\r"; 

/* ========================= Test for Convergence ========================== */

        notConverged = true;  // Assume we have not converged yet
        
        // Get the average of the last LOOKBEHIND_AMNT iterations 
        bufferAvg = avgOf<int, LOOKBEHIND_AMNT>(historyBuffer);

        // If this iteration is close to the average 
        if (testRMSE > bufferAvg)
        {   
            notConverged = false;  // We have converged to suitable weights
        }    
        
        // Else, record this iteration's inaccuracy to check against later
        if (bufferCounter == LOOKBEHIND_AMNT) { bufferCounter = 0; } 
        historyBuffer[bufferCounter] = testRMSE;
        
        bufferCounter++;
        numIter++;
    }

//============================= Summary of Run ============================
    // --- Generate Confusion Matrix ---
    confusionMatrix.zero();
    for (int i = 0; i < NUM_DATA; i++)
    {
        outVec = network.classify(inputData[i]); 
        int expectedClass = classOf<float,OUTPUT_DIM>(expectedOutput[i]);
        int outputClass = classOf<float,OUTPUT_DIM>(outVec);
        confusionMatrix[outputClass][expectedClass] += 1;
    }

    //cout << " Num Iterations: " << numIter << "\n";
    /*cout << " ========== SUMMARY OF RUN ========== " << endl << endl;

    cout << " Num Iterations: " << numIter << "\n";
    
    cout << " - Initial Weight Matrices - " << endl;
    intialWeights_ih.show();
    intialWeights_ho.show();

    cout << " - Final Weight Matrices - " << endl;
    network.getWeights_ih().show();
    network.getWeights_ho().show(); 

    */// Print Confusion Matrix
    cout << "\n - Confusion Matrix: - \n";
    for (int i = 0; i < OUTPUT_DIM; i++)
    {
        for (int j = 0; j < OUTPUT_DIM; j++)
        {
            cout << setw(3) << confusionMatrix[i][j] << " "; 
        }
        cout << "\n";
    }
    cout << "\n";
    
    // Print Recall
    cout.precision(3);
    for (int i = 0; i < OUTPUT_DIM; i++)
    {
        float sumRow = 0;
        float sumCol = 0;
        for (int j = 0; j < OUTPUT_DIM; j++)
        {
            sumRow += confusionMatrix[i][j];
            sumCol += confusionMatrix[j][i];
        }
        cout << "Recall " << i << ":    ";
        if (sumRow == 0) { cout << 0 << endl; } else { 
        cout << confusionMatrix[i][i] / sumRow << endl; }
        cout << "Precision " << i << ": ";
        if (sumCol == 0) { cout << 0 << endl; } else {
        cout << confusionMatrix[i][i] / sumCol << endl << endl; }
    }
    cout << endl;
/*
    cout << " - Classifications of all Datapoints - \n";
    cout << " Datapoint | OutputClass | Actual Class" << endl;
    for (int i = 0; i < NUM_DATA; i++)
    {
        outVec = network.classify(inputData[i]); 
        int expectedClass = classOf<float,OUTPUT_DIM>(expectedOutput[i]);
        int outputClass = classOf<float,OUTPUT_DIM>(outVec);
        cout << setw(3) << i+1 << " | ";
        cout << classOf(outputClass) << " | ";
        cout << classOf(expectedClass) << "\n";
    }*/
}

//============================= Utilitiy Functions ============================

template<class T, int n>
float avgOf(array<T,n> arr)
{
    float sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }
    return sum / n;
}

template<class T,int n,int m> 
Matrix<T,n,m> readCSV(string filename) 
{
    ifstream      file(filename);
   string        delimiter = ",";
    size_t        pos       = 0;
    int           j,i       = 0;
    string        token;
    Matrix<T,n,m> temp;
    
    //getline(file,token);
    for (string line; getline(file, line); )
    {
        j = 0;
        while ((pos = line.find(delimiter)) != std::string::npos) 
        {
            token = line.substr(0, pos);
            temp[i][j] = stof(token.c_str());
            line.erase(0, pos + delimiter.length());
            j++;
        }
        temp[i][j] = stof(line.c_str());
        i++;
    }
    return temp;
}

template<class T, int n, int m, int k>
Matrix<T,n,m> preprocess(Matrix<T,n,k> raw)
{
    Matrix<T,n,m> processed;
    array<T,m>    maxs;
    array<T,m>    mins;

    for (int i = 0; i < m; i++)
    {
        maxs[i] = raw[0][i+1];
        mins[i] = raw[0][i+1];
    }

    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (raw[i][j+1] > maxs[j]) { maxs[j] = raw[i][j+1]; }
            if (raw[i][j+1] < mins[j]) { mins[j] = raw[i][j+1]; }
        }
    } 

    for (int i = 0; i < n; i++)
    {
       for (int j = 0; j < m; j++)
       {
            processed[i][j] = (raw[i][j+1] - mins[j]) / (maxs[j] - mins[j]);
       } 
    }
    return processed;
}

template<class T, int n, int m>
array<int,NUM_CLASSES+1> getClassIndices(Matrix<T,n,m> rawData)
{
    int counter = 1;
    array<int,NUM_CLASSES+1> indices = {0};
    
    for (int i = 1; i < n; i++)
    {
        if (rawData[i][m-1] != rawData[i-1][m-1])
        {
            indices[counter] = i;
            counter++;
        }
    }
    indices[NUM_CLASSES] = n;
    return indices;
}
template<class T, int n> float rmse(array<T,n> output, array<T,n> expected)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += (output[i] - expected[i]) * (output[i] - expected[i]);
    }
    return pow(err / n, 0.5);
}

template<class T, int n, int m> 
Matrix<T,n,m> getExpectedVectors(Matrix<float,NUM_DATA,RAW_DATA_DIM> raw)
{
    Matrix<T,n,m> expected;
    for (int i = 0; i < n; i++)
    {
        if(raw[i][RAW_DATA_DIM-1] > IS_POP_THRESH)
        {
            expected[i][0] = 1;
            expected[i][1] = 0;
        }
        else
        {
            expected[i][0] = 0;
            expected[i][1] = 1;
        }
    }
    return expected;
}

template<class T, int n>
int classOf(array<T,n> vec)
{
    int indexMax = 0;
    float max = vec[0];
    for (int i = 1; i < n; i++)
    {
        if (vec[i] > max)
        {
            max = vec[i];
            indexMax = i;
        }
    }    
    return indexMax;
}

void displayHeader()
{
    cout << "--- Running Assignment 2 Network ---\n";
    cout << "\nParameters:\n";
    cout << "\tHidden Nodes:    " << NUM_HIDDEN << "\n"; 
    cout << "\tLearning Rate:   " << LEARNING_RATE << "\n";
    cout << "\tMax Iterations:  " << MAX_ITER << "\n";
    cout << "\tMomentum Rate:   " << MOMENTUM_RATE << "\n";
    cout << "\t\% used to train: " << TRAIN_TEST_RATIO << "\n";
    cout << "\n";

}

int classOf(int x)
{
   int cl = x + 1; 
   if (cl > 3) { cl++; }
   return cl;
}

