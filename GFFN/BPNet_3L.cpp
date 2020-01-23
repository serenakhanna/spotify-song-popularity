#include "Matrix.cpp"
#include <cmath>

template< class T, int n> array<T,n> signum(array<T,n>);

template< class T, unsigned int i, unsigned int h, unsigned int o>
class BPNet_3L
{
    private:
        Matrix<T,h,i> weights_ih;  // Weights from input to hidden layer
        Matrix<T,o,h> weights_ho;  // Weights from hidden to output layer

        array<T,h> momentum_ih;
        array<T,o> momentum_ho;

    public:
        const float LEARN_RATE;
        const float MOMENTUM_RATE;
        
        BPNet_3L(float, float);
        
        void          show() const;
        array<T,o>    classify(array<T,i>) const;
        void          train(array<T,i>, array<T,o>);
        Matrix<T,h,i> getWeights_ih() const;
        Matrix<T,o,h> getWeights_ho() const;
};

// --- Member function declarations ---

template< class T, unsigned int i, unsigned int h, unsigned int o>
BPNet_3L<T,i,h,o>::BPNet_3L(float k, float alpha) : LEARN_RATE(k), 
                                                    MOMENTUM_RATE(alpha) 
{
    weights_ih = Matrix<T,h,i>::randomMatrix(1);
    weights_ho = Matrix<T,o,h>::randomMatrix(1);

    momentum_ih = {0};
    momentum_ho = {0};
}

template< class T, unsigned int i, unsigned int h, unsigned int o>
void BPNet_3L<T,i,h,o>::show() const
{
    weights_ih.show();
    weights_ho.show();
} 

template< class T, unsigned int i, unsigned int h, unsigned int o>
array<T,o> BPNet_3L<T,i,h,o>::classify(array<T,i> input) const
{
    array<T,h> h_activation = signum<T,h>(weights_ih.mvproduct(input));
    return signum<T,o>(weights_ho.mvproduct(h_activation));
}

template< class T, unsigned int i, unsigned int h, unsigned int o>
void BPNet_3L<T,i,h,o>::train(array<T,i> input, array<T,o> expected)
{
    // Compute outputs at hidden and then output nodes
    array<T,h> h_out = signum<T,h>(weights_ih.mvproduct(input));
    array<T,o> o_out = signum<T,o>(weights_ho.mvproduct(h_out));
    
    // Compute the weight change for the weights from hidden to output layer
    array<T,o> delta_ho;
    for (int j = 0; j < o; j++) 
    {
        // delta    = error                    * derivative of output function
        delta_ho[j] = (expected[j] - o_out[j]) * o_out[j] * (1 - o_out[j]);
    }
     
    // Compute the weight change for the weights from input to hidden layer
    array<T,h> delta_ih;
    for (int j = 0; j < h; j++)
    {
        T sum = 0;
        for (int k = 0; k < o; k++)  // Propagate error from all output nodes
        {
            sum += delta_ho[k] * weights_ho[k][j];
        }
        // delta    = error * derivative of output function
        delta_ih[j] = sum * h_out[j] * (1 - h_out[j]);
    }
    
    // Update hidden-output layer weights
    for (int j = 0; j < h; j++)
    {
        weights_ho.addToColumn(j, h_out[j] * LEARN_RATE, delta_ho);
        weights_ho.addToColumn(j, MOMENTUM_RATE, momentum_ho);
    }
     
    // Update input-hidden layer weights
    for (int j = 0; j < i; j++)
    {
        weights_ih.addToColumn(j, input[j] * LEARN_RATE, delta_ih);
        weights_ih.addToColumn(j, MOMENTUM_RATE, momentum_ih);
    }

    momentum_ho = delta_ho;
    momentum_ih = delta_ih;
}

template< class T, unsigned int i, unsigned int h, unsigned int o>
Matrix<T,h,i> BPNet_3L<T,i,h,o>::getWeights_ih() const
{
    return weights_ih;
}

template< class T, unsigned int i, unsigned int h, unsigned int o>
Matrix<T,o,h> BPNet_3L<T,i,h,o>::getWeights_ho() const
{
    return weights_ho;
}

// --- non-member declarations --- 

template<class T, int n>
array<T,n> signum(array<T,n> input)
{
    array<T,n> activation;
    for (int i = 0; i < n; i++) {
        activation[i] = 1.0 / (1.0 + exp(-input[i]));
    }
    return activation;

}

