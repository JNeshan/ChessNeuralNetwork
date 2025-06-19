#ifndef LAYER_H
#define LAYER_H

class Layer {
public:
    Layer();
    virtual ~Layer();

    virtual void forward() = 0;
    virtual void backward() = 0;
};

#endif // LAYER_H