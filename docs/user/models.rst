Loading and Building Models
===========================

The main entry point is :func:`qalma.alpsmodels.model_from_alps_xml`, which reads an ALPS XML file and returns a `SystemDescriptor` object.

.. autofunction:: qalma.alpsmodels.model_from_alps_xml

Once loaded, you can build system objects via:

.. autofunction:: qalma.model.build_system

.. autoclass:: qalma.model.SystemDescriptor
   :members:
   :inherited-members:
