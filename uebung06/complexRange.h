/*
* complexRange.h
*
*  Created on: 17.12.2014
*      Author: raub
*/

#ifndef COMPLEXRANGE_H_
#define COMPLEXRANGE_H_

#include <iostream>

struct CRange
{
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float xstep;
    float ystep;
    dim3 window;

    CRange()
{
    xmin = 0.0f;
    ymin = 0.0f;
    xmax = 0.0f;
    ymax = 0.0f;
    window.x = 0.0f;
    window.y = 0.0f;
}

CRange(float x1, float y1, float x2, float y2, dim3 windowSize)
{
    set(x1, y1, x2, y2, windowSize);
}

void set(float x1, float y1, float x2, float y2, dim3 windowSize)
{
    xmin = x1;
    ymin = y1;
    xmax = x2;
    ymax = y2;

    xstep = (xmax - xmin) / (float)windowSize.x;
    ystep = (ymax - ymin) / (float)windowSize.y;

    window = windowSize;
}

void set(dim3 windowSize)
{
    xstep = (xmax - xmin) / (float)windowSize.x;
    ystep = (ymax - ymin) / (float)windowSize.y;

    window = windowSize;
}

void set(float x1, float y1, float x2, float y2)
{
    xmin = x1;
    ymin = y1;
    xmax = x2;
    ymax = y2;
}
};

#endif /* COMPLEXRANGE_H_ */
