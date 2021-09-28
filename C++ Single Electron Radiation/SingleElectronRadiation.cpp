#include <iostream>
#include <Eigen/Dense>

inline constexpr double eCharge{ 1.602176634e-19 }; // Coulombs
inline constexpr double vacPerm{ 8.8541878128e-12 }; // Farads per metre
inline constexpr double pi{ 3.141592653589793238462643383279502884 };
inline constexpr int c{ 299792458 }; // speed of light, m/s


double RelativisticTimeDelay(Eigen::Vector3d evaluationPosition,Eigen::Vector3d ePosition) {
    Eigen::Vector3d evaluationElectronSeparation{ (evaluationPosition-ePosition) };
    double separationMagnitude{ evaluationElectronSeparation.norm() };
    double timeDelay{ separationMagnitude/c };
    return timeDelay;
}

Eigen::Vector3d RelFarEField(Eigen::Vector3d evaluationPosition,double Time,Eigen::Vector3d ePosition,Eigen::Vector3d eVelocity,Eigen::Vector3d eAcceleration) {
    double premultiplier{ eCharge/(4*pi*vacPerm*c) };
    Eigen::Vector3d evaluationElectronSeparation{ (evaluationPosition-ePosition) };
    double separationMagnitude{ evaluationElectronSeparation.norm() };
    Eigen::Vector3d normalizedEvaluationElectronSeparation{ evaluationElectronSeparation };
    normalizedEvaluationElectronSeparation.normalize();
    premultiplier = premultiplier * 1 / pow((1 - normalizedEvaluationElectronSeparation.dot(eVelocity) / c),3);

    Eigen::Vector3d farFieldPart{ normalizedEvaluationElectronSeparation.cross(
                                ( normalizedEvaluationElectronSeparation-eVelocity/c).cross(eAcceleration/c)
                                ) / separationMagnitude };
    Eigen::Vector3d relFarEField{ premultiplier*farFieldPart };

    return relFarEField;
}


Eigen::Vector3d RelNearEField(Eigen::Vector3d evaluationPosition,double Time,Eigen::Vector3d ePosition,Eigen::Vector3d eVelocity,Eigen::Vector3d eAcceleration) {
    double premultiplier{ eCharge/(4*pi*vacPerm) };
    Eigen::Vector3d evaluationElectronSeparation{ (evaluationPosition-ePosition) };
    double separationMagnitude{ evaluationElectronSeparation.norm() };
    Eigen::Vector3d normalizedEvaluationElectronSeparation{ evaluationElectronSeparation };
    normalizedEvaluationElectronSeparation.normalize();
    premultiplier = premultiplier * 1 / pow((1 - normalizedEvaluationElectronSeparation.dot(eVelocity) / c),3);

    Eigen::Vector3d nearFieldPart{ (1-pow(eVelocity.norm()/c,2))*(normalizedEvaluationElectronSeparation-eVelocity/c) / pow(separationMagnitude,2) };
    Eigen::Vector3d relNearEField{ premultiplier*nearFieldPart };

    return relNearEField;
}

double PoyntingVectorMagnitude(double electricFieldMagnitude) {
    double poyntingMagnitude{vacPerm*c*pow(electricFieldMagnitude,2)};
    return poyntingMagnitude;
}


int main() {
    // Testing functions with a random point in the middle of a bathtub trajectory
    Eigen::Vector3d detectorPosition{ 0.05,0.0,0.0 };
    double time{ 6.733512428092677019e-07 };
    Eigen::Vector3d ePosition{ 3.291256444075408281e-04,1.446536715994627684e-04,-7.509693298748249674e-03 };
    Eigen::Vector3d eVelocity{ 5.324891768002842367e+07,5.790348564830546826e+07,-1.373440678852058016e+06 };
    Eigen::Vector3d eAcceleration{ -1.063518797840980582e+19,8.841211371036892160e+18,3.178407561331115246e+09 };
    Eigen::Vector3d relFarEField{ RelFarEField(detectorPosition, time, ePosition, eVelocity, eAcceleration) };
    std::cout << "Far field:\n" << relFarEField << std::endl;

    Eigen::Vector3d relNearEField{ RelNearEField(detectorPosition, time, ePosition, eVelocity, eAcceleration) };
    std::cout << "Near field:\n" << relNearEField << std::endl;

    Eigen::Vector3d totalEField{ relFarEField+relNearEField };
    double poyntingMagnitude{ PoyntingVectorMagnitude(totalEField.norm()) };
    std::cout << "Poynting magnitude:\n" << poyntingMagnitude << std::endl;
    return 0;
}