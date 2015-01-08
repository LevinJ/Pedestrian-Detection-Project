/***************************************************************
\Author:	Qingxiong Yang (http://vision.ai.uiuc.edu/~qyang6/)
\Function:	Basic functions.
****************************************************************/
#ifndef QX_BASIC_H
#define QX_BASIC_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <stdexcept>

#include <string>
#include <memory.h>


#define QX_DEF_PADDING 0

using std::max;
using std::min;


class qx_timer
{
public:
    void start();
    float stop();
    void time_display(char *disp="",int nr_frame=1);
    void fps_display(char *disp="",int nr_frame=1);
private:
    clock_t m_begin;
    clock_t m_end;
};

/// Simpler and more general memory allocation

template<typename T>
T* qx_alloc_1d(int d1, int padding = QX_DEF_PADDING)
{
    T* a = new T[d1 + padding];
    T* p = a;
    return p;
}

template<typename T>
void qx_free_1d(T* p)
{
    if(p) // not really necessary but consistent with others
    {
        delete[] p;
    }
}

template<typename T>
T** qx_alloc_2d(int d1, int d2, int padding = QX_DEF_PADDING)
{
    T* a=0;
    T** p=0;

    try
    {
        a = new T[d1*d2 + padding];
        p = new T*[d1];
    }
    catch(...)
    {
        delete[] a;
        delete[] p;
        throw;
    }

    for(int i1=0; i1<d1; ++i1)
    {
        p[i1] = a + i1*d2;
    }

    return p;
}

template<typename T>
void qx_free_2d(T** p)
{
    if(p)
    {
        delete[] p[0];
        delete[] p;
    }
}

template<typename T>
T*** qx_alloc_3d(int d1, int d2, int d3, int padding = QX_DEF_PADDING)
{
    T* a = 0;
    T** p = 0;
    T*** pp = 0;

    try
    {
        a = new T[d1*d2*d3 + padding];
        p = new T*[d1*d2];
        pp = new T**[d1];
    }
    catch(...)
    {
        delete[] a;
        delete[] p;
        delete[] pp;
        throw;
    }

    for(int i1=0; i1<d1; ++i1)
    {
        pp[i1] = p + i1 * d2;
        for(int i2=0; i2<d2; ++i2)
        {
            pp[i1][i2] = a + (i1*d2 + i2)*d3;
        }
    }
    return pp;
}

template<typename T>
void qx_free_3d(T*** p)
{
    if(p)
    {
        delete[] p[0][0];
        delete[] p[0];
        delete[] p;
    }
}

template<typename T>
T**** qx_alloc_4d(int d1, int d2, int d3, int d4, int padding = QX_DEF_PADDING)
{
    T* a = 0;
    T** p = 0;
    T*** pp = 0;
    T**** ppp = 0;

    try
    {
        a = new T[d1*d2*d3*d4 + padding];
        p = new T*[d1*d2*d3];
        pp = new T**[d1*d2];
        ppp = new T***[d1];
    }
    catch(...)
    {
        delete[] a;
        delete[] p;
        delete[] pp;
        delete[] ppp;
        throw;
    }

    for(int i1=0; i1<d1; ++i1)
    {
        ppp[i1] = pp + i1 * d2;
        for(int i2=0; i2<d2; ++i2)
        {
            ppp[i1][i2] = p + (i1*d2 + i2)*d3;
            for(int i3=0; i3<d3; ++i3)
            {
               ppp[i1][i2][i3] = a + ((i1*d2 + i2)*d3 + i3)*d4;
            }
        }
    }
    return ppp;
}

template<typename T>
void qx_free_4d(T**** p)
{
    if(p)
    {
        delete[] p[0][0][0];
        delete[] p[0][0];
        delete[] p[0];
        delete[] p;
    }
}

// Exception-safe wrappers
template<typename T>
class qx_array_1d
{
    public:
        qx_array_1d()
            : m_d1(0),
              m_padding(0),
              m_arr(0)
        {}

        explicit qx_array_1d(int d1, int padding=QX_DEF_PADDING)
            : m_d1(d1),
              m_padding(padding),
              m_arr(qx_alloc_1d<T>(m_d1, m_padding))
        {}
        
        qx_array_1d(const qx_array_1d& a)
            : m_d1(a.m_d1),
              m_padding(a.m_padding),
              m_arr(a.m_arr ? qx_alloc_1d<T>(m_d1, m_padding) : 0)
        {
            if(a.m_arr)
                std::copy(a.m_arr, a.m_arr + m_d1 + m_padding, m_arr);
        }

        ~qx_array_1d() { qx_free_1d<T>(m_arr); }

        qx_array_1d& operator= (const qx_array_1d& a)
        {
            qx_array_1d(a).swap(*this);
            return *this;
        }

        void swap(qx_array_1d& a)
        {
            using std::swap;
            swap(a.m_d1, m_d1);
            swap(a.m_padding, m_padding);
            swap(a.m_arr, m_arr);
        }

        typedef T * type;
        typedef const T * const_type;
        operator type() { return m_arr; }
        operator const_type() const { return m_arr; }

    private:
        int m_d1;
        int m_padding;
        T* m_arr;
};

template<typename T>
class qx_array_2d
{
    public:
        qx_array_2d()
            : m_d1(0),
              m_d2(0),
              m_padding(0),
              m_arr(0)
        {}

        qx_array_2d(int d1, int d2, int padding=QX_DEF_PADDING)
            : m_d1(d1),
              m_d2(d2),
              m_padding(padding),
              m_arr(qx_alloc_2d<T>(m_d1, m_d2, m_padding))
        {}
        
        qx_array_2d(const qx_array_2d& a)
            : m_d1(a.m_d1),
              m_d2(a.m_d2),
              m_padding(a.m_padding),
              m_arr(a.m_arr ? qx_alloc_2d<T>(m_d1, m_d2, m_padding) : 0)
        {
            if(a.m_arr)
                std::copy(a.m_arr[0], a.m_arr[0] + m_d1*m_d2 + m_padding, m_arr[0]);
        }

        ~qx_array_2d() { qx_free_2d<T>(m_arr); }

        qx_array_2d& operator= (const qx_array_2d& a)
        {
            qx_array_2d(a).swap(*this);
            return *this;
        }

        void swap(qx_array_2d& a)
        {
            using std::swap;
            swap(a.m_d1, m_d1);
            swap(a.m_d2, m_d2);
            swap(a.m_padding, m_padding);
            swap(a.m_arr, m_arr);
        }

        typedef T * const * type;
        typedef const T * const * const_type;
        operator type () { return m_arr; }
        operator const_type () const { return m_arr; }

    private:
        int m_d1;
        int m_d2;
        int m_padding;
        T** m_arr;
};

template<typename T>
class qx_array_3d
{
    public:
        qx_array_3d()
            : m_d1(0),
              m_d2(0),
              m_d3(0),
              m_padding(0),
              m_arr(0)
        {}

        qx_array_3d(int d1, int d2, int d3, int padding=QX_DEF_PADDING)
            : m_d1(d1),
              m_d2(d2),
              m_d3(d3),
              m_padding(padding),
              m_arr(qx_alloc_3d<T>(m_d1, m_d2, m_d3, m_padding))
        {}
        
        qx_array_3d(const qx_array_3d& a)
            : m_d1(a.m_d1),
              m_d2(a.m_d2),
              m_d3(a.m_d3),
              m_padding(a.m_padding),
              m_arr(a.m_arr ? qx_alloc_3d<T>(m_d1, m_d2, m_d3, m_padding) : 0)
        {
            if(a.m_arr)
                std::copy(a.m_arr[0][0], a.m_arr[0][0] + m_d1*m_d2*m_d3 + m_padding, m_arr[0][0]);
        }

        ~qx_array_3d() { qx_free_3d<T>(m_arr); }

        qx_array_3d& operator= (const qx_array_3d& a)
        {
            qx_array_3d(a).swap(*this);
            return *this;
        }

        void swap(qx_array_3d& a)
        {
            using std::swap;
            swap(a.m_d1, m_d1);
            swap(a.m_d2, m_d2);
            swap(a.m_d3, m_d3);
            swap(a.m_padding, m_padding);
            swap(a.m_arr, m_arr);
        }

        typedef T * const * const * type;
        typedef const T * const * const * const_type;
        operator type () { return m_arr; }
        operator const_type () const { return m_arr; }

    private:
        int m_d1;
        int m_d2;
        int m_d3;
        int m_padding;
        T*** m_arr;
};

template<typename T>
class qx_array_4d
{
    public:
        qx_array_4d()
            : m_d1(0),
              m_d2(0),
              m_d3(0),
              m_d4(0),
              m_padding(0),
              m_arr(0)
        {}

        qx_array_4d(int d1, int d2, int d3, int d4, int padding=QX_DEF_PADDING)
            : m_d1(d1),
              m_d2(d2),
              m_d3(d3),
              m_d4(d4),
              m_padding(padding),
              m_arr(qx_alloc_4d<T>(m_d1, m_d2, m_d3, m_d4, m_padding))
        {}
        
        qx_array_4d(const qx_array_4d& a)
            : m_d1(a.m_d1),
              m_d2(a.m_d2),
              m_d3(a.m_d3),
              m_d4(a.m_d4),
              m_padding(a.m_padding),
              m_arr(a.m_arr ? qx_alloc_4d<T>(m_d1, m_d2, m_d3, m_d4, m_padding) : 0)
        {
            if(a.m_arr)
                std::copy(a.m_arr[0][0][0], a.m_arr[0][0][0] + m_d1*m_d2*m_d3*m_d4 + m_padding, m_arr[0][0][0]);
        }

        ~qx_array_4d() { qx_free_4d<T>(m_arr); }

        qx_array_4d& operator= (const qx_array_4d& a)
        {
            qx_array_4d(a).swap(*this);
            return *this;
        }

        void swap(qx_array_4d& a)
        {
            using std::swap;
            swap(a.m_d1, m_d1);
            swap(a.m_d2, m_d2);
            swap(a.m_d3, m_d3);
            swap(a.m_d4, m_d4);
            swap(a.m_padding, m_padding);
            swap(a.m_arr, m_arr);
        }

        typedef T * const * const * const * type;
        typedef const T * const * const * const * const_type;
        operator type() { return m_arr; }
        operator const_type() const { return m_arr; }

    private:
        int m_d1;
        int m_d2;
        int m_d3;
        int m_d4;
        int m_padding;
        T**** m_arr;
};

/// Memory allocation (legacy)
inline double** qx_allocd(int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_2d<double>(r,c,padding);
}

inline void qx_freed(double **p)
{
    qx_free_2d<double>(p);
}

inline double *** qx_allocd_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_3d<double>(n,r,c,padding);
}

inline void qx_freed_3(double ***p)
{
    qx_free_3d<double>(p);
}

inline double**** qx_allocd_4(int t,int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_4d<double>(t,n,r,c,padding);
}

inline void qx_freed_4(double ****p)
{
    qx_free_4d<double>(p);
}

inline unsigned char** qx_allocu(int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_2d<unsigned char>(r,c,padding);
}

inline void qx_freeu(unsigned char **p)
{
    qx_free_2d<unsigned char>(p);
}

inline unsigned char *** qx_allocu_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_3d<unsigned char>(n,r,c,padding);
}

inline void qx_freeu_3(unsigned char ***p)
{
    qx_free_3d<unsigned char>(p);
}

inline unsigned char**** qx_allocu_4(int t,int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_4d<unsigned char>(t,n,r,c,padding);
}

inline void qx_freeu_4(unsigned char ****p)
{
    qx_free_4d<unsigned char>(p);
}

inline unsigned short** qx_allocus(int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_2d<unsigned short>(r,c,padding);
}

inline void qx_freeus(unsigned short **p)
{
    qx_free_2d<unsigned short>(p);
}

inline unsigned short *** qx_allocus_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_3d<unsigned short>(n,r,c,padding);
}

inline void qx_freeus_3(unsigned short ***p)
{
    qx_free_3d<unsigned short>(p);
}

inline char** qx_allocc(int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_2d<char>(r,c,padding);
}

inline void qx_freec(char **p)
{
    qx_free_2d<char>(p);
}

inline char *** qx_allocc_3(int n, int r, int c, int padding=QX_DEF_PADDING)
{
    return qx_alloc_3d<char>(n,r,c,padding);
}

inline void qx_freec_3(char ***p)
{
    qx_free_3d<char>(p);
}

inline short** qx_allocs(int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_2d<short>(r,c,padding);
}

inline void qx_frees(short **p)
{
    qx_free_2d<short>(p);
}

inline short*** qx_allocs_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_3d<short>(n,r,c,padding);
}

inline void qx_frees_3(short ***p)
{
    qx_free_3d<short>(p);
}

inline short**** qx_allocs_4(int t,int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_4d<short>(t,n,r,c,padding);
}

inline void qx_frees_4(short ****p)
{
    qx_free_4d<short>(p);
}

inline float** qx_allocf(int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_2d<float>(r,c,padding);
}

inline void qx_freef(float **p)
{
    qx_free_2d<float>(p);
}

inline float *** qx_allocf_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_3d<float>(n,r,c,padding);
}

inline void qx_freef_3(float ***p)
{
    qx_free_3d<float>(p);
}

inline float**** qx_allocf_4(int t,int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_4d<float>(t,n,r,c,padding);
}

inline void qx_freef_4(float ****p)
{
    qx_free_4d<float>(p);
}

inline int** qx_alloci(int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_2d<int>(r,c,padding);
}

inline void qx_freei(int **p)
{
    qx_free_2d<int>(p);
}

inline int*** qx_alloci_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_3d<int>(n,r,c,padding);
}

inline void qx_freei_3(int ***p)
{
    qx_free_3d<int>(p);
}

inline int**** qx_alloci_4(int t,int n,int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_4d<int>(t,n,r,c,padding);
}

inline void qx_freei_4(int ****p)
{
    qx_free_4d<int>(p);
}

inline long** qx_allocli(int r,int c,int padding=QX_DEF_PADDING)
{
    return qx_alloc_2d<long>(r,c,padding);
}

inline void qx_freeli(long **p)
{
    qx_free_2d<long>(p);
}
#endif // QX_BASIC_H
