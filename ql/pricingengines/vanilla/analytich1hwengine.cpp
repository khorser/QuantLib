/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2012 Klaus Spanderen

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <https://www.quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file analytichestonengine.cpp
    \brief analytic Heston-Hull-White engine based on the H1-HW approximation
*/

#include <ql/math/distributions/gammadistribution.hpp>
#include <ql/math/integrals/gaussianquadratures.hpp>
#include <ql/pricingengines/vanilla/analytich1hwengine.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>

namespace QuantLib {

    namespace {

        /* E[sqrt(v_t)] for the Heston variance started at v0:
           v_t/c(t) is non-central chi-squared with d degrees of freedom
           and non-centrality lambda(t), so the expectation is a Poisson
           mixture of central ones. The weights are summed outwards from
           the mode of the Poisson distribution, so that no term overflows.
           Beyond lambda = 1e5 the asymptotic approximation avoids
           summing thousands of terms. */
        Real varianceRootMean(Real v0, Real kappa, Real theta, Real sigma,
                              Time t) {
            if (t <= 0.0)
                return std::sqrt(v0);
            const Real e = std::exp(-kappa*t);
            if (sigma == 0.0)
                return std::sqrt(theta + (v0-theta)*e);

            const Real oneMinusE = -std::expm1(-kappa*t);
            const Real c = sigma*sigma/(4.0*kappa)*oneMinusE;
            const Real lambda = 4.0*kappa*v0*e/(sigma*sigma*oneMinusE);
            const Real d = 4.0*kappa*theta/(sigma*sigma);
            if (lambda >= 1.0e5)
                return std::sqrt(c*(lambda-1.0)
                                 + c*d*(1.0 + 1.0/(2.0*(d+lambda))));

            const GammaFunction g;
            const Real h = 0.5*lambda, a = 0.5*(d+1.0), b = 0.5*d;
            const Real m = std::floor(h);
            const Real w0 = std::exp(-h + (m > 0.0 ? m*std::log(h) : 0.0)
                                     - g.logValue(m+1.0));
            const Real shape = b+m;
            const Real inverseShape = 1.0/shape;
            // Gamma(x + 1/2) / Gamma(x) = sqrt(x) *
            // (1 - 1/(8x) + 1/(128x^2) + 5/(1024x^3) + O(x^-4)).
            const Real r0 = shape >= 1.0e5
                ? std::sqrt(shape)*(1.0 + inverseShape*(-0.125 + inverseShape*
                    (1.0/128.0 + inverseShape*5.0/1024.0)))
                : 1.0/boost::math::tgamma_delta_ratio(shape, 0.5);
            const Real tiny = 1.0e-17;

            Real sum = w0*r0, w = w0, r = r0, k = m;
            for (;;) {
                w *= h/(k+1.0);
                r *= (a+k)/(b+k);
                k += 1.0;
                const Real term = w*r;
                sum += term;
                if (term <= tiny*sum)
                    break;
            }
            w = w0;
            r = r0;
            k = m;
            while (k > 0.0) {
                w *= k/h;
                r *= (b+k-1.0)/(a+k-1.0);
                k -= 1.0;
                const Real term = w*r;
                sum += term;
                if (term <= tiny*sum)
                    break;
            }
            return std::sqrt(2.0*c)*sum;
        }

    }

    // integration helper class
    class AnalyticH1HWEngine::Fj_Helper {

      public:
        Fj_Helper(const Handle<HestonModel>& hestonModel,
                  const ext::shared_ptr<HullWhite>& hullWhiteModel,
                  Real rho_xr, Time term, Real strike, Size j);

        std::complex<Real> operator()(Real u) const;

      private:
        Real c(Time t) const;
        Real lambda(Time t) const;
        Real Lambda(Time t) const;
        Real LambdaApprox(Time t) const;

        const Size j_;
        const Real lambda_, eta_;
        const Real v0_, kappa_, theta_, gamma_;
        const Real d_;
        const Real rhoSr_;
        const Time term_;
    };

    AnalyticH1HWEngine::Fj_Helper::Fj_Helper(
        const Handle<HestonModel>& hestonModel,
        const ext::shared_ptr<HullWhite>& hullWhiteModel,
        Real rhoSr, Time term, Real, Size j)
    : j_     (j),
      lambda_(hullWhiteModel->a()),
      eta_   (hullWhiteModel->sigma()),
      v0_    (hestonModel->v0()),
      kappa_ (hestonModel->kappa()),
      theta_ (hestonModel->theta()),
      gamma_ (hestonModel->sigma()),
      d_     (4.0*kappa_*theta_/(gamma_*gamma_)),
      rhoSr_ (rhoSr),
      term_  (term) {
    }

    Real AnalyticH1HWEngine::Fj_Helper::c(Time t) const {
        return gamma_*gamma_/(4*kappa_)*(1.0-std::exp(-kappa_*t));
    }

    Real AnalyticH1HWEngine::Fj_Helper::lambda(Time t) const {
        return  4.0*kappa_*v0_*std::exp(-kappa_*t)
               /(gamma_*gamma_*(1.0-std::exp(-kappa_*t)));
    }

    Real AnalyticH1HWEngine::Fj_Helper::LambdaApprox(Time t) const {
        return std::sqrt( c(t)*(lambda(t)-1.0)
                        + c(t)*d_*(1.0 + 1.0/(2.0*(d_+lambda(t)))));
    }

    Real AnalyticH1HWEngine::Fj_Helper::Lambda(Time t) const {
        const GammaFunction g = GammaFunction();
        const Size maxIter = 1000;
        const Real lambdaT = lambda(t);

        Size i=0;
        Real retVal = 0.0, s;

        do {
            Real k = static_cast<Real>(i);
            s=std::exp(k*std::log(0.5*lambdaT) + g.logValue(0.5*(1+d_)+k)
                        - g.logValue(k+1) - g.logValue(0.5*d_+k));
            retVal += s;
        } while (s > std::numeric_limits<float>::epsilon() && ++i < maxIter);

        QL_REQUIRE(i < maxIter, "can not calculate Lambda");

        retVal *= std::sqrt(2*c(t)) * std::exp(-0.5*lambdaT);
        return retVal;
    }

    std::complex<Real> AnalyticH1HWEngine::Fj_Helper::operator()(Real u) const {

        const Real gamma2 = gamma_*gamma_;

        Real a, b, c;
        if (8.0*kappa_*theta_/gamma2 > 1.0) {
            a = std::sqrt(theta_-gamma2/(8.0*kappa_));
            b = std::sqrt(v0_) - a;
            c =-std::log((LambdaApprox(1.0)-a)/b);
        }
        else {
            a = std::sqrt(gamma2/(2.0*kappa_))
                *std::exp(  GammaFunction().logValue(0.5*(d_+1.0))
                          - GammaFunction().logValue(0.5*d_));

            const Time t1 = 0.0;
            const Time t2 = 1.0/kappa_;

            const Real Lambda_t1 = std::sqrt(v0_);
            const Real Lambda_t2 = Lambda(t2);

            c = std::log((Lambda_t2-a)/(Lambda_t1-a))/(t1-t2);
            b = std::exp(c*t1)*(Lambda_t1-a);
        }

        QL_REQUIRE(std::isfinite(c) && c != lambda_,
                   "the fitted approximation of E[sqrt(v)] has no solution "
                   "for these Heston parameters (c = " << c << "); "
                   "use VarianceRootMean::Exact");

        const std::complex<Real> I4 =
            -1.0 / lambda_ * std::complex<Real>(u * u, ((j_ == 1U) ? -u : u)) *
            (b / c * (1.0 - std::exp(-c * term_)) + a * term_ +
             a / lambda_ * (std::exp(-lambda_ * term_) - 1.0) +
             b / (c - lambda_) * std::exp(-c * term_) * (1.0 - std::exp(-term_ * (lambda_ - c))));

        return eta_*rhoSr_*I4;
    }


    AnalyticH1HWEngine::AnalyticH1HWEngine(
        const ext::shared_ptr<HestonModel>& model,
        const ext::shared_ptr<HullWhite>& hullWhiteModel,
        Real rhoSr, Size integrationOrder,
        VarianceRootMean mean, Size meanIntegrationOrder)
    : AnalyticHestonHullWhiteEngine(model, hullWhiteModel, integrationOrder),
      rhoSr_(rhoSr), mean_(mean),
      meanIntegrationOrder_(meanIntegrationOrder) {
        QL_REQUIRE(rhoSr_ >= 0.0, "Fourier integration is not stable if "
                    "the equity interest rate correlation is negative");
        QL_REQUIRE(meanIntegrationOrder_ > 0,
                   "the mean integration order must be positive");
    }

    AnalyticH1HWEngine::AnalyticH1HWEngine(
        const ext::shared_ptr<HestonModel>& model,
        const ext::shared_ptr<HullWhite>& hullWhiteModel,
        Real rhoSr, Real relTolerance, Size maxEvaluations,
        VarianceRootMean mean, Size meanIntegrationOrder)
    : AnalyticHestonHullWhiteEngine(model, hullWhiteModel,
                                    relTolerance, maxEvaluations),
      rhoSr_(rhoSr), mean_(mean),
      meanIntegrationOrder_(meanIntegrationOrder) {
        QL_REQUIRE(rhoSr_ >= 0.0, "Fourier integration is not stable if "
                    "the equity interest rate correlation is negative");
        QL_REQUIRE(meanIntegrationOrder_ > 0,
                   "the mean integration order must be positive");
    }

    void AnalyticH1HWEngine::update() {
        cached_ = false;
        AnalyticHestonHullWhiteEngine::update();
    }

    Real AnalyticH1HWEngine::exactMeanIntegral(Time t) const {
        // Cache int_0^t E[sqrt(v_s)] (1 - exp(-a (t-s))) / a ds.
        // The scaled kernel tends to t-s as a tends to zero.
        const Real v0 = model_->v0(), kappa = model_->kappa(),
                   theta = model_->theta(), sigma = model_->sigma(),
                   a = hullWhiteModel_->a();
        if (!(cached_ && t == cachedT_ && v0 == cachedV0_
              && kappa == cachedKappa_ && theta == cachedTheta_
              && sigma == cachedSigma_ && a == cachedA_)) {
            const GaussLegendreIntegration quadrature(meanIntegrationOrder_);
            cachedIntegral_ = 0.5*t*quadrature([&](Real x) {
                const Time s = 0.5*t*(x+1.0);
                const Time tau = t-s;
                const Real z = a*tau;
                const Real kernel = tau*(z == 0.0 ? 1.0 : -std::expm1(-z)/z);
                return varianceRootMean(v0, kappa, theta, sigma, s)*kernel;
            });
            cachedT_ = t;
            cachedV0_ = v0;
            cachedKappa_ = kappa;
            cachedTheta_ = theta;
            cachedSigma_ = sigma;
            cachedA_ = a;
            cached_ = true;
        }
        return cachedIntegral_;
    }

    std::complex<Real> AnalyticH1HWEngine::addOnTerm(Real u, Time t, Size j)
    const {
        if (mean_ == VarianceRootMean::FittedExponential)
            return AnalyticHestonHullWhiteEngine::addOnTerm(u, t, j)
                + Fj_Helper(model_, hullWhiteModel_, rhoSr_, t, 0.0, j)(u);

        const Real eta = hullWhiteModel_->sigma();
        const std::complex<Real> I4 =
            -std::complex<Real>(u * u, ((j == 1U) ? -u : u))
            * exactMeanIntegral(t);
        return AnalyticHestonHullWhiteEngine::addOnTerm(u, t, j)
            + eta*rhoSr_*I4;
    }
}

