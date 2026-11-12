import numpy as np

import porepy as pp


class CapillaryPressure(pp.PorePyModel):
    """Mixin for capillary pressure using Brooks-Corey model."""

    def entry_pressure(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Return the cellwise entry pressure for the facies.

        """
        # Teller totalt antall celler i domenet:
        size = sum(domain.num_cells for domain in domains)
        # setter enrty pressure for de to sandlaga (setter samme entry pressure som for facies 1 og 2 i spe11a)
        p_entry_lower = float(self.params.get("p_entry_lower", 300.0))
        p_entry_upper = float(self.params.get("p_entry_upper", 1500.0))

        values = np.zeros(size) #en tom vektor som skal fylles opp med entry pressure for hver celle
        start = 0
        is_upper_layer = getattr(self, "øvre_lag")
        for domain in domains:
            is_upper = is_upper_layer(domain) #finner hvilke celelr som ligger i øvre lag
            values[start : start + domain.num_cells] = np.where(
                is_upper, p_entry_upper, p_entry_lower
            ) #setter riktig pressure i hver celle.
            start += domain.num_cells

        return pp.wrap_as_dense_ad_array(values, size, name="entry_pressure")

    def data_to_export(self):
        """Add the cellwise entry-pressure field to ParaView output."""
        data = super().data_to_export()
        subdomains = self.mdg.subdomains()
        entry_pressure = self.equation_system.evaluate(
            self.entry_pressure(subdomains)
        )
        capillary_pressure = self.equation_system.evaluate(
            self.capillary_pressure(subdomains)
        )
        offsets = np.cumsum([0] + [domain.num_cells for domain in subdomains])

        for index, domain in enumerate(subdomains):
            data.append(
                (
                    domain,
                    "entry_pressure",
                    entry_pressure[offsets[index] : offsets[index + 1]],
                )
            )
            data.append(
                (
                    domain,
                    "capillary_pressure",
                    capillary_pressure[offsets[index] : offsets[index + 1]],
                )
            )
        return data

    def capillary_pressure(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Beregn kapillartrykk fra våtfasemetningen med Brooks-Corey-modellen.

        Metoden beregner den konstitutive relasjonen
        ``p_c(s_w) = p_n - p_w``. Den returnerer altså kapillartrykket som
        funksjon av metningen; den introduserer ikke et nytt vanntrykk.
        """
        
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        # Hente metning av wetting phase:
        aqueous = self.fluid.phases[0]
        sat = aqueous.saturation(domains)
        # Hente immobile metning for wetting phase:
        sw_imm = float(self.params["sw_imm"])
        # Beregn normalisert metning for den mobile delen av wetting phase:
        mobile_saturation_interval = max(1.0 - sw_imm, 1e-12)
        sat_norm = (sat - pp.ad.Scalar(sw_imm)) / pp.ad.Scalar(
            mobile_saturation_interval
        )
        # passe på at normalisert metning ikke blir negativ (se ligning 5 i beskrivelsen)
        sat_eff = f_max(sat_norm, pp.ad.Scalar(0.0))
        # må også legge på minimum så ikke pc blir uendelig når sw <= sw,imm
        minimum = pp.ad.Function(
            lambda value: pp.ad.maximum(value, 1e-12), "minimum_saturation"
        )
        sat_eff = minimum(sat_eff)
        c2 = float(self.params["c2"]) #fra problem def
        # Beregne brooks-corey cappress til slutt (ligning 8 i problem def):
        pcap = self.entry_pressure(domains) * (
            sat_eff ** pp.ad.Scalar(-1.0 / c2)
        )
        
        # Keep the Brooks-Corey singularity from dominating the nonlinear solve
        # when the wetting saturation approaches its residual value.
        pcap_max = float(self.params.get("pcap_max", np.inf))
        if np.isfinite(pcap_max):
            capillary_maximum = pp.ad.Function(
                lambda value: pp.ad.maximum(value, -pcap_max),
                "negative_capillary_maximum",
            )
            pcap = -capillary_maximum(-pcap)

        pcap.set_name("capillary_pressure")
        return pcap

    