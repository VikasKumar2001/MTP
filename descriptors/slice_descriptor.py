class SliceDescriptor:
    """
    Milestone 3/8: Extracts structural slice signatures.
    """
    def extract(self, slicing_data: dict) -> dict:
        # Retrieves the normalized 20-value slice signature
        signature = slicing_data.get("slice_signature", [])
        
        return {
            "slice_vector": signature
        }