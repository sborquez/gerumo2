# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# From Detectron2 framework:
# https://github.com/facebookresearch/detectron2/tree/7801ac3d595a0663d16c0c9a6a339c77b2ddbdfb/detectron2/config

"""
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction,
and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by
the copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all
other entities that control, are controlled by, or are under common
control with that entity. For the purposes of this definition,
"control" means (i) the power, direct or indirect, to cause the
direction or management of such entity, whether by contract or
otherwise, or (ii) ownership of fifty percent (50%) or more of the
outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity
exercising permissions granted by this License.

"Source" form shall mean the preferred form for making modifications,
including but not limited to software source code, documentation
source, and configuration files.

"Object" form shall mean any form resulting from mechanical
transformation or translation of a Source form, including but
not limited to compiled object code, generated documentation,
and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or
Object form, made available under the License, as indicated by a
copyright notice that is included in or attached to the work
(an example is provided in the Appendix below).

"Derivative Works" shall mean any work, whether in Source or Object
form, that is based on (or derived from) the Work and for which the
editorial revisions, annotations, elaborations, or other modifications
represent, as a whole, an original work of authorship. For the purposes
of this License, Derivative Works shall not include works that remain
separable from, or merely link (or bind by name) to the interfaces of,
the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including
the original version of the Work and any modifications or additions
to that Work or Derivative Works thereof, that is intentionally
submitted to Licensor for inclusion in the Work by the copyright owner
or by an individual or Legal Entity authorized to submit on behalf of
the copyright owner. For the purposes of this definition, "submitted"
means any form of electronic, verbal, or written communication sent
to the Licensor or its representatives, including but not limited to
communication on electronic mailing lists, source code control systems,
and issue tracking systems that are managed by, or on behalf of, the
Licensor for the purpose of discussing and improving the Work, but
excluding communication that is conspicuously marked or otherwise
designated in writing by the copyright owner as "Not a Contribution."

"Contributor" shall mean Licensor and any individual or Legal Entity
on behalf of whom a Contribution has been received by Licensor and
subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
this License, each Contributor hereby grants to You a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable
copyright license to reproduce, prepare Derivative Works of,
publicly display, publicly perform, sublicense, and distribute the
Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
this License, each Contributor hereby grants to You a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable
(except as stated in this section) patent license to make, have made,
use, offer to sell, sell, import, and otherwise transfer the Work,
where such license applies only to those patent claims licensable
by such Contributor that are necessarily infringed by their
Contribution(s) alone or by combination of their Contribution(s)
with the Work to which such Contribution(s) was submitted. If You
institute patent litigation against any entity (including a
cross-claim or counterclaim in a lawsuit) alleging that the Work
or a Contribution incorporated within the Work constitutes direct
or contributory patent infringement, then any patent licenses
granted to You under this License for that Work shall terminate
as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
Work or Derivative Works thereof in any medium, with or without
modifications, and in Source or Object form, provided that You
meet the following conditions:

(a) You must give any other recipients of the Work or
Derivative Works a copy of this License; and

(b) You must cause any modified files to carry prominent notices
stating that You changed the files; and

(c) You must retain, in the Source form of any Derivative Works
that You distribute, all copyright, patent, trademark, and
attribution notices from the Source form of the Work,
excluding those notices that do not pertain to any part of
the Derivative Works; and

(d) If the Work includes a "NOTICE" text file as part of its
distribution, then any Derivative Works that You distribute must
include a readable copy of the attribution notices contained
within such NOTICE file, excluding those notices that do not
pertain to any part of the Derivative Works, in at least one
of the following places: within a NOTICE text file distributed
as part of the Derivative Works; within the Source form or
documentation, if provided along with the Derivative Works; or,
within a display generated by the Derivative Works, if and
wherever such third-party notices normally appear. The contents
of the NOTICE file are for informational purposes only and
do not modify the License. You may add Your own attribution
notices within Derivative Works that You distribute, alongside
or as an addendum to the NOTICE text from the Work, provided
that such additional attribution notices cannot be construed
as modifying the License.

You may add Your own copyright statement to Your modifications and
may provide additional or different license terms and conditions
for use, reproduction, or distribution of Your modifications, or
for any such Derivative Works as a whole, provided Your use,
reproduction, and distribution of the Work otherwise complies with
the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
any Contribution intentionally submitted for inclusion in the Work
by You to the Licensor shall be under the terms and conditions of
this License, without any additional terms or conditions.
Notwithstanding the above, nothing herein shall supersede or modify
the terms of any separate license agreement you may have executed
with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
names, trademarks, service marks, or product names of the Licensor,
except as required for reasonable and customary use in describing the
origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
agreed to in writing, Licensor provides the Work (and each
Contributor provides its Contributions) on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied, including, without limitation, any warranties or conditions
of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE. You are solely responsible for determining the
appropriateness of using or redistributing the Work and assume any
risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
whether in tort (including negligence), contract, or otherwise,
unless required by applicable law (such as deliberate and grossly
negligent acts) or agreed to in writing, shall any Contributor be
liable to You for damages, including any direct, indirect, special,
incidental, or consequential damages of any character arising as a
result of this License or out of the use or inability to use the
Work (including but not limited to damages for loss of goodwill,
work stoppage, computer failure or malfunction, or any and all
other commercial damages or losses), even if such Contributor
has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
the Work or Derivative Works thereof, You may choose to offer,
and charge a fee for, acceptance of support, warranty, indemnity,
or other liability obligations and/or rights consistent with this
License. However, in accepting such obligations, You may act only
on Your own behalf and on Your sole responsibility, not on behalf
of any other Contributor, and only if You agree to indemnify,
defend, and hold each Contributor harmless for any liability
incurred by, or claims asserted against, such Contributor by reason
of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

To apply the Apache License to your work, attach the following
boilerplate notice, with the fields enclosed by brackets "[]"
replaced with your own identifying information. (Don't include
the brackets!)  The text should be enclosed in the appropriate
comment syntax for the file format. We also recommend that a
file or class name and description of purpose be included on the
same "printed page" as the copyright notice for easier
identification within third-party archives.

Copyright [yyyy] [name of copyright owner]


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import inspect
from fvcore.common.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:
    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    .. automethod:: clone
    .. automethod:: freeze
    .. automethod:: defrost
    .. automethod:: is_frozen
    .. automethod:: load_yaml_with_base
    .. automethod:: merge_from_list
    .. automethod:: merge_from_other_cfg
    """

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        """
        Load content from the given config file and merge it into self.
        Args:
            cfg_filename: config filename
            allow_unsafe: allow unsafe yaml syntax
        """
        #assert PathManager.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        from .defaults import _C

        
    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)


global_cfg = CfgNode()


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()


def set_global_cfg(cfg: CfgNode) -> None:
    """
    Let the global config point to the given cfg.
    Assume that the given "cfg" has the key "KEY", after calling
    `set_global_cfg(cfg)`, the key can be accessed by:
    ::
        from detectron2.config import global_cfg
        print(global_cfg.KEY)
    By using a hacky global config, you can access these configs anywhere,
    without having to pass the config object or the values deep into the code.
    This is a hacky feature introduced for quick prototyping / research exploration.
    """
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)


def configurable(init_func=None, *, from_config=None):
    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.
    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}
        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a: cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass
        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite
    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2. It must take `cfg`
            as its first argument.
    """

    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)

            wrapped.from_config = from_config
            return wrapped

        return wrapper


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    from omegaconf import DictConfig

    if len(args) and isinstance(args[0], (_CfgNode, DictConfig)):
        return True
    if isinstance(kwargs.pop("cfg", None), (_CfgNode, DictConfig)):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False