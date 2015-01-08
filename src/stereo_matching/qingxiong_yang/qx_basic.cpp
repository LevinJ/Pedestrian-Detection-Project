#include "qx_basic.hpp"


void qx_timer::start()
{
    m_begin=clock();
}
float qx_timer::stop()
{
    m_end=clock();
    return ( float(m_end-m_begin)/CLOCKS_PER_SEC );
}
void qx_timer::time_display(char *disp,int nr_frame)
{
    printf("Running time (%s) is: %5.5f Seconds.\n",disp,stop()/nr_frame);
}
void qx_timer::fps_display(char *disp,int nr_frame)
{
    printf("Running time (%s) is: %5.5f frame per second.\n",disp,(float)nr_frame/stop());
}
