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

/*! \file analytich1hwengine.hpp
    \brief analytic Heston-Hull-White engine based on the H1-HW approximation
*/

#ifndef quantlib_analytic_h1_hw_engine_hpp
#define quantlib_analytic_h1_hw_engine_hpp

#include <ql/models/equity/hestonmodel.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/pricingengines/vanilla/analytichestonhullwhiteengine.hpp>

namespace QuantLib {

    //! Analytic Heston-Hull-White engine based on the H1-HW approximation
    /*! This class is pricing a european option under the following process

        \f[
        \begin{array}{rcl}
        dS(t, S)  &=& (r-d) S dt +\sqrt{v} S dW_1 \\
        dv(t, S)  &=& \kappa (\theta - v) dt + \sigma \sqrt{v} dW_2 \\
        dr(t)     &=& (\theta(t) - a r) dt + \eta dW_3 \\
        dW_1 dW_2 &=& \rho_{S,v} dt, \rho_{S,r} >= 0 \\
        dW_1 dW_3 &=& \rho_{S.r} dt \\
        dW_2 dW_3 &=& 0 dt \\
        \end{array}
        \f]
    */

    /*! References:

        Lech A. Grzelak, Cornelis W. Oosterlee,
        On The Heston Model with Stochastic,
        http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1382902

        Lech A. Grzelak,
        Equity and Foreign Exchange Hybrid Models for
        Pricing Long-Maturity Financial Derivatives,
        http://repository.tudelft.nl/assets/uuid:a8e1a007-bd89-481a-aee3-0e22f15ade6b/PhDThesis_main.pdf

        \ingroup vanillaengines

        \test the correctness of the returned value is tested by
              reproducing results available in web/literature, testing
              against QuantLib's analytic Heston,
              the Black-Scholes-Merton Hull-White engine and
              the finite difference Heston-Hull-White engine
    */

    /*! The approximation replaces \f$ \sqrt{v_s} \f$ in the equity/rate
        cross term by its expectation. By default (FittedExponential) that
        expectation is itself approximated by \f$ a + b e^{-cs} \f$, as in
        the reference, with \f$ c \f$ fitted at \f$ s = 1 \f$ or
        \f$ s = 1/\kappa \f$, depending on the variance parameters. The fit
        can fail where the expectation is not monotone (for instance
        \f$ v_0 \f$ close to \f$ \theta \f$), and the engine then fails.
        Exact integrates the expectation by Gauss-Legendre quadrature on
        meanIntegrationOrder nodes, using an asymptotic approximation at
        large noncentrality.
    */
    class AnalyticH1HWEngine : public AnalyticHestonHullWhiteEngine {
      public:
        enum class VarianceRootMean { FittedExponential, Exact };

        AnalyticH1HWEngine(const ext::shared_ptr<HestonModel>& model,
                           const ext::shared_ptr<HullWhite>& hullWhiteModel,
                           Real rhoSr, Size integrationOrder = 144,
                           VarianceRootMean mean = VarianceRootMean::FittedExponential,
                           Size meanIntegrationOrder = 64);

        AnalyticH1HWEngine(const ext::shared_ptr<HestonModel>& model,
                           const ext::shared_ptr<HullWhite>& hullWhiteModel,
                           Real rhoSr, Real relTolerance, Size maxEvaluations,
                           VarianceRootMean mean = VarianceRootMean::FittedExponential,
                           Size meanIntegrationOrder = 64);

        void update() override;

      protected:
        std::complex<Real> addOnTerm(Real phi, Time t, Size j) const override;

      private:
        class Fj_Helper;

        Real exactMeanIntegral(Time t) const;

        const Real rhoSr_;
        const VarianceRootMean mean_;
        const Size meanIntegrationOrder_;

        // the exact-mean integral does not depend on the Fourier node,
        // so it is kept for the last maturity and model parameters
        mutable bool cached_ = false;
        mutable Time cachedT_ = 0.0;
        mutable Real cachedV0_ = 0.0, cachedKappa_ = 0.0, cachedTheta_ = 0.0,
                     cachedSigma_ = 0.0, cachedA_ = 0.0, cachedIntegral_ = 0.0;
    };
}

#endif
