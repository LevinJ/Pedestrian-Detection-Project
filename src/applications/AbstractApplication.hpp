#ifndef ABSTRACTAPPLICATION_HPP
#define ABSTRACTAPPLICATION_HPP

namespace doppia {


/// Abstract class of an application
/// @see BaseApplication for a real application skeleton
class AbstractApplication
{
public:

    virtual int main(int argc, char *argv[]) = 0;
};

} // end of namespace doppia

#endif // ABSTRACTAPPLICATION_HPP
