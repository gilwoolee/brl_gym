import xml.etree.ElementTree as ET
from warnings import warn
import numpy as np


class MujocoUpdater(object):
    ignore_params = {
        'name',
        'type',
        'axisangle',
        'axis',
        'armature',
        'rgba',
        'range',
        # 'pos',
        'quat',
        'ref',
        'limited',
        'margin',
        'material',
        'solimplimit',
        'solreflimit',
        'stiffness',
        # 'friction',
        'damping'
    }

    def __init__(self, xml_string):
        self.xml_root = ET.fromstring(xml_string)
        self._xml_string = xml_string

    def get_body_names(self):
        return [child.attrib['name'] for child in self.xml_root.iter('body')]

    def get_body(self, body_name):
        for child in self.xml_root.iter('body'):
            if child.attrib['name'] == body_name:
                return child
        raise ValueError('Body {0} not found'.format(body_name))

    def _get_body_element(self, body_name, element_name):
        assert element_name in ('geom', 'joint')

        body = self.get_body(body_name)
        element = body.find(element_name)
        default = self.xml_root.find('default').find(element_name)

        if element is None and default is None:
            warn('Body {0} does not have {1}'.format(body_name, element_name))
        elif element is not None and default is not None:
            for key, val in default.attrib.items():
                if key not in element.attrib and key not in self.ignore_params:
                    element.attrib[key] = val
        return element

    def get_geom(self, body_name):
        return self._get_body_element(body_name, 'geom')

    def get_joint(self, body_name):
        return self._get_body_element(body_name, 'joint')

    def _get_params(self, element_name):
        assert element_name in ('body', 'geom', 'joint')

        params = {}
        for body_name in self.get_body_names():
            element = None
            if element_name == 'body':
                element = self.get_body(body_name)
            elif element_name == 'geom':
                element = self.get_geom(body_name)
            elif element_name == 'joint':
                element = self.get_joint(body_name)
            if element is None:
                continue

            for key, value in element.attrib.items():
                if key in self.ignore_params:
                    continue
                try:
                    value = np.array([float(x) for x in value.split()])
                except ValueError:
                    # Ignore any non-numeric values
                    continue

                param_name = '{body_name}__{element_name}__{key}'.format(
                    body_name=body_name, element_name=element_name, key=key)
                params[param_name] = value

        return params

    def get_body_params(self):
        return self._get_params('body')

    def get_geom_params(self):
        return self._get_params('geom')

    def get_joint_params(self):
        return self._get_params('joint')

    def get_params(self):
        body_params = self.get_body_params()
        geom_params = self.get_geom_params()
        joint_params = self.get_joint_params()
        all_params = {**body_params, **geom_params, **joint_params}
        return all_params

    def set_param(self, element_name, body_name, param_name, param_value):
        assert element_name in ('body', 'geom', 'joint')

        element = None
        if element_name == 'body':
            element = self.get_body(body_name)
        elif element_name == 'geom':
            element = self.get_geom(body_name)
        elif element_name == 'joint':
            element = self.get_joint(body_name)
        if element is None:
            raise ValueError('Element {0} not found in body {1}'.format(element_name, body_name))

        prev_value = element.get(param_name)
        if prev_value is None:
            warn('Body {0} {1} does not have {2}'.format(body_name, element_name, param_name))

        element.set(param_name, str(param_value))

    def set_body_param(self, body_name, body_param_name, body_param_value):
        return self.set_param('body', body_name, body_param_name, body_param_value)

    def set_geom_param(self, body_name, geom_param_name, geom_param_value):
        return self.set_param('geom', body_name, geom_param_name, geom_param_value)

    def set_joint_param(self, body_name, joint_param_name, joint_param_value):
        return self.set_param('joint', body_name, joint_param_name, joint_param_value)

    @staticmethod
    def set_params(xml_string, params):
        updater = MujocoUpdater(xml_string)

        for k, v in params.items():
            body_name, param_type, param_name = k.split('__')
            v = ' '.join([str(x) for x in v])
            updater.set_param(param_type, body_name, param_name, v)

        return ET.tostring(updater.xml_root, encoding='unicode')
